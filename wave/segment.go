package wave

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var (
	// ErrSegmentInvalidFormat is returned by NewSegmentRecorder when the
	// sample type, frame rate, or channel count is invalid.
	ErrSegmentInvalidFormat = errors.New("segment recorder format is invalid")

	// ErrSegmentInvalidDuration is returned when the requested segment
	// duration is zero or negative.
	ErrSegmentInvalidDuration = errors.New("segment duration must be greater than zero")

	// ErrSegmentDurationTooSmall is returned when the requested duration is
	// positive but represents less than one audio frame at the configured
	// frame rate (e.g. one nanosecond at 44100 Hz).
	ErrSegmentDurationTooSmall = errors.New("segment duration is shorter than a single audio frame")

	// ErrSegmentDurationTooLong is returned when a single segment could not be
	// represented as a wave file (the 'data' chunk size is a 32-bit field).
	// Note that the recording as a whole is not limited; only the configured
	// segment length is.
	ErrSegmentDurationTooLong = errors.New("segment duration is too long to represent as a wave file")

	// ErrSegmentPartialFrame is returned by Finish when the input ended in the
	// middle of an audio frame. Whole frames are preserved (and the segments
	// containing them remain valid), but the trailing partial frame is not
	// written.
	ErrSegmentPartialFrame = errors.New("input ended in the middle of an audio frame")

	// ErrSegmentFinished is returned when samples are written after Finish has
	// been called.
	ErrSegmentFinished = errors.New("segment recorder has already been finished")

	// ErrSegmentCancelled is returned when samples are written after the
	// recorder has been cancelled, and by Finish on a cancelled recorder.
	ErrSegmentCancelled = errors.New("segment recorder was cancelled")

	// ErrSegmentFormatMismatch is returned by WriteBlock when the declared
	// format differs from the format fixed at construction time.
	ErrSegmentFormatMismatch = errors.New("block format does not match the segment recorder format")
)

// SegmentFormat describes the audio format used by every segment in a
// recording. A recorder's format is fixed at construction time, so all
// segments produced by one recorder are directly comparable and can be
// concatenated without conversion.
type SegmentFormat struct {
	SampleType   SampleType `json:"sampleType"`
	FrameRate    uint32     `json:"frameRate"`
	ChannelCount uint16     `json:"channelCount"`
}

// BlockAlign returns the number of bytes in a single audio frame.
func (f SegmentFormat) BlockAlign() int {
	return f.SampleType.Size() * int(f.ChannelCount)
}

func (f SegmentFormat) validate() error {
	if !f.SampleType.IsValid() {
		return fmt.Errorf("%w: invalid sample type: %d", ErrSegmentInvalidFormat, int(f.SampleType))
	}
	if f.FrameRate == 0 {
		return fmt.Errorf("%w: frame rate must be greater than zero", ErrSegmentInvalidFormat)
	}
	if f.ChannelCount == 0 {
		return fmt.Errorf("%w: channel count must be greater than zero", ErrSegmentInvalidFormat)
	}
	return nil
}

// SegmentInfo describes a completed segment within the manifest.
type SegmentInfo struct {
	// Index is the 1-based position of this segment within the recording.
	Index int `json:"index"`

	// FileName is the base name of the .wav file, relative to the manifest's
	// directory.
	FileName string `json:"fileName"`

	// StartFrame is the position of the first audio frame in this segment,
	// relative to the beginning of the recording (zero-based). A frame
	// contains one sample per channel.
	StartFrame uint64 `json:"startFrame"`

	// StartSample is the position of the first sample in this segment,
	// relative to the beginning of the recording (zero-based). It equals
	// StartFrame * Format.ChannelCount.
	StartSample uint64 `json:"startSample"`

	// FrameCount is the number of complete audio frames in this segment.
	FrameCount uint64 `json:"frameCount"`

	// SampleCount is FrameCount * Format.ChannelCount.
	SampleCount uint64 `json:"sampleCount"`

	// DataBytes is the number of bytes in the segment's 'data' chunk (not
	// including the wave header or the RIFF padding byte).
	DataBytes uint64 `json:"dataBytes"`
}

// SegmentManifest is the on-disk index of a segmented recording. It is
// rewritten atomically after every completed segment, so at any point in time
// it describes exactly those segments that were successfully flushed to disk.
type SegmentManifest struct {
	// Version identifies the manifest schema. It is currently always 1.
	Version int `json:"version"`

	// Format describes the audio format shared by all segments.
	Format SegmentFormat `json:"format"`

	// TargetSegmentDurationSeconds is the requested segment duration.
	TargetSegmentDurationSeconds float64 `json:"targetSegmentDurationSeconds"`

	// FramesPerSegment is the number of frames in every segment except the
	// last (which may be shorter).
	FramesPerSegment uint64 `json:"framesPerSegment"`

	// TotalFrames and TotalSamples summarize the completed segments.
	TotalFrames  uint64 `json:"totalFrames"`
	TotalSamples uint64 `json:"totalSamples"`

	// Segments lists only segments that have been completely written and
	// flushed. Failed or in-progress segments are never included.
	Segments []SegmentInfo `json:"segments"`

	// Complete is true only if Finish was called successfully.
	Complete bool `json:"complete"`

	// Cancelled is true if Cancel (or a cancelled context) stopped the
	// recording.
	Cancelled bool `json:"cancelled,omitempty"`
}

// segmentFile is the subset of *os.File operations used by a
// SegmentRecorder. It is an interface to allow tests to inject short writes
// and disk failures.
type segmentFile interface {
	io.WriteSeeker
	Sync() error
	Close() error
}

type segmentRecorderOptions struct {
	manifestFileName string
	segmentPrefix    string
	ctx              context.Context
}

// SegmentRecorderOption is a functional argument used as part of
// NewSegmentRecorder.
type SegmentRecorderOption func(*segmentRecorderOptions) error

// WithManifestFileName overrides the default manifest file name
// ("manifest.json").
func WithManifestFileName(name string) SegmentRecorderOption {
	return func(opts *segmentRecorderOptions) error {
		if name == "" {
			return errors.New("manifest file name must not be empty")
		}
		opts.manifestFileName = name
		return nil
	}
}

// WithSegmentFileNamePrefix overrides the default segment file prefix
// ("segment-"). Segment files are named "<prefix>000001.wav",
// "<prefix>000002.wav", and so on. The numeric field expands beyond six digits
// when needed.
func WithSegmentFileNamePrefix(prefix string) SegmentRecorderOption {
	return func(opts *segmentRecorderOptions) error {
		if prefix == "" {
			return errors.New("segment file name prefix must not be empty")
		}
		opts.segmentPrefix = prefix
		return nil
	}
}

// WithContext associates a context with the recorder. If the context is
// cancelled, the next write (or Finish) cancels the recording in the same way
// as Cancel: the in-progress segment is removed, completed segments are kept,
// and the manifest is updated.
func WithContext(ctx context.Context) SegmentRecorderOption {
	return func(opts *segmentRecorderOptions) error {
		if ctx == nil {
			return errors.New("context must not be nil")
		}
		opts.ctx = ctx
		return nil
	}
}

// A SegmentRecorder writes a long-running sample stream to a sequence of
// .wav files, splitting the stream only on complete audio frame boundaries.
// Each segment is an independent, standards-compliant wave file that can be
// opened with NewReader, and a JSON manifest names the segments and records
// their start positions, frame counts, and the shared audio format.
//
// Segments are rotated when they reach the number of complete frames that
// corresponds to the requested duration (rounded down). The final segment may
// be shorter than the target duration. Segments are finalized (header flushed,
// file synced, file closed, manifest atomically updated) before the next one
// is started, so a later failure never invalidates earlier segments.
//
// Input is not required to arrive frame-aligned: WriteRaw and the typed
// WriteXXX methods buffer any trailing partial frame until the rest of the
// frame is supplied. If the stream ends mid-frame, Finish preserves all whole
// frames but returns ErrSegmentPartialFrame.
//
// A SegmentRecorder is safe for concurrent use, though writes are serialized.
// The zero value is not usable; call NewSegmentRecorder.
type SegmentRecorder struct {
	mu sync.Mutex

	dir      string
	format   SegmentFormat
	manifest string
	prefix   string
	ctx      context.Context

	framesPerSegment uint64
	duration         time.Duration

	// Injection points used by tests. They default to the real filesystem.
	openFile func(path string) (segmentFile, error)
	remove   func(path string) error

	// Streaming state
	carry   []byte // bytes belonging to an incomplete frame
	current *segmentState

	// Completed segments and cumulative counters. 'completedFrames' counts
	// frames in finalized segments only; 'totalFrames' additionally includes
	// the in-progress segment.
	segments        []SegmentInfo
	completedFrames uint64
	totalFrames     uint64

	terminalErr error
	finished    bool
	cancelled   bool
}

type segmentState struct {
	path   string
	file   segmentFile
	writer *Writer
	frames uint64
}

// NewSegmentRecorder creates a recorder that writes .wav segments of
// approximately 'segmentDuration' each into 'dir'. The duration is converted
// to a whole number of frames (rounded down) and must be at least one frame
// but small enough that each segment's 'data' chunk fits in a wave file.
//
// The directory is created if it does not exist. Existing segment files are
// never overwritten: the recorder uses exclusive file creation and returns an
// error if a target name already exists.
func NewSegmentRecorder(
	dir string,
	format SegmentFormat,
	segmentDuration time.Duration,
	opts ...SegmentRecorderOption,
) (*SegmentRecorder, error) {

	if dir == "" {
		return nil, errors.New("output directory must not be empty")
	}
	if err := format.validate(); err != nil {
		return nil, err
	}
	if segmentDuration <= 0 {
		return nil, ErrSegmentInvalidDuration
	}

	options := &segmentRecorderOptions{
		manifestFileName: "manifest.json",
		segmentPrefix:    "segment-",
	}
	for _, opt := range opts {
		if err := opt(options); err != nil {
			return nil, err
		}
	}

	// The manifest and segment names must be plain file names (not paths) so
	// that all output stays within the requested directory. The manifest also
	// must not be able to collide with a generated segment name, since the
	// manifest is replaced by rename.
	for _, name := range []string{options.manifestFileName, options.segmentPrefix} {
		if name != filepath.Base(name) || name == "." || name == ".." {
			return nil, fmt.Errorf("output file name must not contain a directory: %q", name)
		}
	}
	if strings.HasPrefix(options.manifestFileName, options.segmentPrefix) {
		return nil, fmt.Errorf(
			"manifest file name %q must not begin with the segment prefix %q",
			options.manifestFileName, options.segmentPrefix,
		)
	}

	framesPerSegment, err := framesForDuration(segmentDuration, format.FrameRate)
	if err != nil {
		return nil, err
	}

	// A wave file stores the size of its 'data' chunk in a uint32 field, and
	// the enclosing RIFF chunk size must fit in a uint32 as well. Reject
	// segment durations that could not be represented. This does not limit the
	// total recording length, since each segment is a separate file.
	maxDataBytes := maxSegmentDataBytes(format)
	if framesPerSegment > maxDataBytes/uint64(format.BlockAlign()) {
		return nil, ErrSegmentDurationTooLong
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create output directory: %w", err)
	}

	// Refuse to write over an existing recording. Otherwise the initial
	// manifest write would destroy the previous recording's index before the
	// first (exclusive) segment creation failed.
	manifestPath := filepath.Join(dir, options.manifestFileName)
	if _, err := os.Lstat(manifestPath); err == nil {
		return nil, fmt.Errorf("output directory already contains a manifest (%s)", manifestPath)
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("inspect output directory: %w", err)
	}

	r := &SegmentRecorder{
		dir:              dir,
		format:           format,
		manifest:         options.manifestFileName,
		prefix:           options.segmentPrefix,
		ctx:              options.ctx,
		framesPerSegment: framesPerSegment,
		duration:         segmentDuration,
		segments:         make([]SegmentInfo, 0, 4),
	}
	r.openFile = func(path string) (segmentFile, error) {
		return os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	}
	r.remove = os.Remove

	// Write an initial manifest so that the output directory is immediately
	// self-describing, and so that permission problems are reported up front
	// rather than after the first segment's worth of audio has been buffered.
	if err := r.writeManifestLocked(false, false); err != nil {
		return nil, err
	}

	return r, nil
}

// Format returns the fixed format used by this recorder.
func (r *SegmentRecorder) Format() SegmentFormat {
	return r.format
}

// Directory returns the output directory.
func (r *SegmentRecorder) Directory() string {
	return r.dir
}

// FramesPerSegment returns the target number of frames per segment.
func (r *SegmentRecorder) FramesPerSegment() uint64 {
	return r.framesPerSegment
}

// FramesWritten returns the total number of complete frames that have been
// written so far, including frames in the in-progress segment.
func (r *SegmentRecorder) FramesWritten() uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.totalFrames
}

// Manifest returns a snapshot of the current manifest. In-progress data is not
// included; only finalized segments appear.
func (r *SegmentRecorder) Manifest() SegmentManifest {
	r.mu.Lock()
	defer r.mu.Unlock()
	complete := r.finished && r.terminalErr == nil && !r.cancelled
	return r.manifestLocked(complete, r.cancelled)
}

// WriteUint8 writes uint8 PCM samples.
func (r *SegmentRecorder) WriteUint8(data []uint8) error {
	if r.format.SampleType != SampleTypeUint8 {
		return ErrWriterExpectedUint8
	}
	encoded := make([]byte, len(data))
	copy(encoded, data)
	return r.WriteRaw(encoded)
}

// WriteInt16 writes int16 PCM samples.
func (r *SegmentRecorder) WriteInt16(data []int16) error {
	if r.format.SampleType != SampleTypeInt16 {
		return ErrWriterExpectedInt16
	}
	encoded := make([]byte, len(data)*2)
	for i, value := range data {
		binary.LittleEndian.PutUint16(encoded[i*2:], uint16(value))
	}
	return r.WriteRaw(encoded)
}

// WriteInt24 writes 24-bit PCM samples (each value stored in an int32).
func (r *SegmentRecorder) WriteInt24(data []int32) error {
	if r.format.SampleType != SampleTypeInt24 {
		return ErrWriterExpectedInt24
	}
	var encoded bytes.Buffer
	encoded.Grow(len(data) * 3)
	if _, err := WritePackedInt24(&encoded, data); err != nil {
		return err
	}
	return r.WriteRaw(encoded.Bytes())
}

// WriteInt32 writes int32 PCM samples.
func (r *SegmentRecorder) WriteInt32(data []int32) error {
	if r.format.SampleType != SampleTypeInt32 {
		return ErrWriterExpectedInt32
	}
	encoded := make([]byte, len(data)*4)
	for i, value := range data {
		binary.LittleEndian.PutUint32(encoded[i*4:], uint32(value))
	}
	return r.WriteRaw(encoded)
}

// WriteFloat32 writes IEEE float32 samples.
func (r *SegmentRecorder) WriteFloat32(data []float32) error {
	if r.format.SampleType != SampleTypeFloat32 {
		return ErrWriterExpectedFloat32
	}
	encoded := make([]byte, len(data)*4)
	for i, value := range data {
		binary.LittleEndian.PutUint32(encoded[i*4:], math.Float32bits(value))
	}
	return r.WriteRaw(encoded)
}

// WriteFloat64 writes IEEE float64 samples.
func (r *SegmentRecorder) WriteFloat64(data []float64) error {
	if r.format.SampleType != SampleTypeFloat64 {
		return ErrWriterExpectedFloat64
	}
	encoded := make([]byte, len(data)*8)
	for i, value := range data {
		binary.LittleEndian.PutUint64(encoded[i*8:], math.Float64bits(value))
	}
	return r.WriteRaw(encoded)
}

// WriteRaw writes already-encoded interleaved samples. The bytes must be
// little-endian and use the SampleType given to NewSegmentRecorder. Inputs do
// not need to contain a whole number of frames; a trailing partial frame is
// buffered until it is completed by a later call.
//
// WriteRaw is useful when samples are already available as bytes (e.g. from a
// device or network stream). Most callers should prefer the typed WriteXXX
// methods.
func (r *SegmentRecorder) WriteRaw(data []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.checkWritableLocked(); err != nil {
		return err
	}
	return r.writeRawLocked(data)
}

// writeRawLocked implements WriteRaw. The caller must hold r.mu and must have
// verified that the recorder accepts writes.
func (r *SegmentRecorder) writeRawLocked(data []byte) error {
	if len(data) == 0 {
		return nil
	}

	// Prepend any buffered bytes from a previously partial frame.
	if len(r.carry) > 0 {
		needed := r.format.BlockAlign() - len(r.carry)
		if len(data) < needed {
			r.carry = append(r.carry, data...)
			return nil
		}
		combined := make([]byte, 0, len(r.carry)+len(data))
		combined = append(combined, r.carry...)
		combined = append(combined, data...)
		data = combined
		r.carry = nil
	}

	whole := len(data) / r.format.BlockAlign() * r.format.BlockAlign()
	if whole < len(data) {
		r.carry = append(r.carry[:0], data[whole:]...)
	}
	if whole == 0 {
		return nil
	}

	return r.writeFramesLocked(data[:whole])
}

// WriteBlock writes a block of already-encoded samples together with an
// explicit description of the format the bytes use. It is intended for
// capture pipelines where the audio device can change its configuration at
// runtime (e.g. the channel count changes when a device is re-opened).
//
// Segments within one recording must share a single format, so a block whose
// format differs from the recorder's format is rejected with
// ErrSegmentFormatMismatch and no data is written. The recorder remains usable;
// the caller can either drop the block or start a new recording for the new
// format. A block that is too short to complete an already-buffered partial
// frame is likewise rejected, since its frame boundary is ambiguous.
func (r *SegmentRecorder) WriteBlock(format SegmentFormat, data []byte) error {
	if err := format.validate(); err != nil {
		return err
	}
	if format != r.format {
		return fmt.Errorf(
			"%w: block is %s/%d Hz/%d ch, recording is %s/%d Hz/%d ch",
			ErrSegmentFormatMismatch,
			format.SampleType, format.FrameRate, format.ChannelCount,
			r.format.SampleType, r.format.FrameRate, r.format.ChannelCount,
		)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.checkWritableLocked(); err != nil {
		return err
	}

	// A buffered partial frame was produced by data of the recorder's own
	// format, so bytes arriving now must complete that frame. This is only
	// ambiguous if they do not. An empty block carries no data and is a no-op.
	if len(data) > 0 && len(r.carry) > 0 && len(data) < r.format.BlockAlign()-len(r.carry) {
		return fmt.Errorf(
			"%w: a partial frame is buffered, so the block's frame boundary is ambiguous",
			ErrSegmentFormatMismatch,
		)
	}

	return r.writeRawLocked(data)
}

// Finish finalizes the in-progress segment (if any) and writes a manifest with
// Complete set to true. Calling Finish more than once is allowed; subsequent
// calls return the same result as the first.
//
// If the input ended in the middle of a frame, all complete frames are still
// written and finalized, the manifest is written with Complete set to false,
// and ErrSegmentPartialFrame is returned.
func (r *SegmentRecorder) Finish() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.cancelled {
		return r.cancelResultLocked()
	}
	if r.terminalErr != nil {
		return r.terminalErr
	}
	if r.finished {
		return nil
	}

	// An already-cancelled context finishes the recording as cancelled; it must
	// not finalize the in-progress segment.
	if err := r.contextErrLocked(); err != nil {
		return err
	}
	r.finished = true

	if len(r.carry) > 0 {
		// Preserve all whole frames that have been received, then report that
		// the final partial frame was dropped.
		r.carry = nil
		if err := r.finalizeLocked(); err != nil {
			return r.failLocked(err)
		}
		if err := r.writeManifestLocked(false, false); err != nil {
			return r.failLocked(err)
		}
		r.terminalErr = ErrSegmentPartialFrame
		return r.terminalErr
	}

	if err := r.finalizeLocked(); err != nil {
		return r.failLocked(err)
	}
	if err := r.writeManifestLocked(true, false); err != nil {
		return r.failLocked(err)
	}
	return nil
}

// Cancel stops the recording. The in-progress segment is closed and removed;
// all previously completed segments are kept, and the manifest is rewritten
// with Cancelled set to true and Complete set to false. Cancel is idempotent.
// Calling Cancel after a successful Finish is a no-op.
func (r *SegmentRecorder) Cancel() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.cancelled {
		return r.cancelResultLocked()
	}
	if r.finished {
		if r.terminalErr != nil {
			return r.terminalErr
		}
		return nil
	}

	r.cancelled = true
	r.carry = nil
	r.discardCurrentLocked()
	if err := r.writeManifestLocked(false, true); err != nil {
		return r.failLocked(err)
	}
	return nil
}

// checkWritableLocked verifies that the recorder can accept more data,
// applying context cancellation as a side effect.
func (r *SegmentRecorder) checkWritableLocked() error {
	// Terminal states take precedence: once the recording has failed or been
	// finished, a later context cancellation must not rewrite its manifest.
	if r.terminalErr != nil {
		return r.terminalErr
	}
	if r.cancelled {
		return ErrSegmentCancelled
	}
	if r.finished {
		return ErrSegmentFinished
	}
	return r.contextErrLocked()
}

func (r *SegmentRecorder) contextErrLocked() error {
	if r.ctx == nil {
		return nil
	}
	select {
	case <-r.ctx.Done():
		if !r.cancelled {
			r.cancelled = true
			r.carry = nil
			r.discardCurrentLocked()
			if err := r.writeManifestLocked(false, true); err != nil {
				return r.failLocked(fmt.Errorf("%w: context cancellation: %v", ErrSegmentCancelled, err))
			}
		}
		return ErrSegmentCancelled
	default:
		return nil
	}
}

// writeFramesLocked appends frame-aligned data, rotating segments as needed.
func (r *SegmentRecorder) writeFramesLocked(data []byte) error {
	blockAlign := r.format.BlockAlign()

	for len(data) > 0 {
		// Notice cancellation between segments so that a single very large
		// write does not have to complete before cancellation takes effect.
		if err := r.contextErrLocked(); err != nil {
			return err
		}

		if r.current == nil {
			if err := r.openSegmentLocked(); err != nil {
				return r.failLocked(err)
			}
		}

		roomFrames := r.framesPerSegment - r.current.frames
		frames := uint64(len(data) / blockAlign)
		if frames > roomFrames {
			frames = roomFrames
		}
		n := int(frames) * blockAlign

		if err := r.current.writer.writeRaw(data[:n]); err != nil {
			return r.failLocked(fmt.Errorf(
				"write segment %d: %w", len(r.segments)+1, err,
			))
		}

		r.current.frames += frames
		r.totalFrames += frames
		data = data[n:]

		if r.current.frames == r.framesPerSegment {
			if err := r.finalizeLocked(); err != nil {
				return r.failLocked(err)
			}
			if err := r.writeManifestLocked(false, false); err != nil {
				return r.failLocked(err)
			}
		}
	}

	return nil
}

func (r *SegmentRecorder) openSegmentLocked() error {
	index := len(r.segments) + 1
	name := fmt.Sprintf("%s%06d.wav", r.prefix, index)
	path := filepath.Join(r.dir, name)

	file, err := r.openFile(path)
	if err != nil {
		// Note: an error from openFile means no usable file was returned
		// (segmentFile may hold a typed nil, so it must not be inspected
		// here). Any partially-created file is the opener's responsibility.
		return fmt.Errorf("create segment %d (%s): %w", index, name, err)
	}

	writer, err := NewWriter(
		file,
		r.format.SampleType,
		r.format.FrameRate,
		WithChannelCount(r.format.ChannelCount),
	)
	if err != nil {
		_ = file.Close()
		_ = r.remove(path)
		return fmt.Errorf("create writer for segment %d (%s): %w", index, name, err)
	}

	r.current = &segmentState{
		path:   path,
		file:   file,
		writer: writer,
	}
	return nil
}

// finalizeLocked flushes, syncs, and closes the in-progress segment, appending
// its metadata to the manifest state. On failure the segment file is removed,
// since it was not fully written and is not recorded in the manifest.
func (r *SegmentRecorder) finalizeLocked() error {
	current := r.current
	if current == nil {
		return nil
	}
	r.current = nil

	err := current.writer.Flush()
	if err == nil {
		err = current.file.Sync()
	}
	closeErr := current.file.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		// The segment is not part of the recording, so its frames must not be
		// counted by FramesWritten.
		r.totalFrames -= current.frames
		_ = r.remove(current.path)
		return fmt.Errorf(
			"finalize segment %d (%s): %w",
			len(r.segments)+1, filepath.Base(current.path), err,
		)
	}

	r.segments = append(r.segments, SegmentInfo{
		Index:       len(r.segments) + 1,
		FileName:    filepath.Base(current.path),
		StartFrame:  r.completedFrames,
		StartSample: r.completedFrames * uint64(r.format.ChannelCount),
		FrameCount:  current.frames,
		SampleCount: current.frames * uint64(r.format.ChannelCount),
		DataBytes:   current.frames * uint64(r.format.BlockAlign()),
	})
	r.completedFrames += current.frames
	return nil
}

// discardCurrentLocked closes and removes the in-progress segment without
// recording it. Frames that were written to it are removed from the running
// total, since they are not part of the finished recording.
func (r *SegmentRecorder) discardCurrentLocked() {
	current := r.current
	if current == nil {
		return
	}
	r.current = nil
	r.totalFrames -= current.frames
	_ = current.file.Close()
	_ = r.remove(current.path)
}

// failLocked transitions the recorder into a terminal error state. The
// in-progress segment is discarded; completed segments are left untouched. A
// best-effort manifest update is attempted so that the on-disk index continues
// to describe exactly the segments that are valid.
func (r *SegmentRecorder) failLocked(err error) error {
	if r.terminalErr == nil {
		r.terminalErr = err
		r.discardCurrentLocked()
		r.carry = nil
		_ = r.writeManifestLocked(false, r.cancelled)
	}
	return err
}

func (r *SegmentRecorder) cancelResultLocked() error {
	// The manifest was written when cancellation occurred. If that write
	// failed, surface the terminal error; otherwise report the cancellation.
	if r.terminalErr != nil {
		return fmt.Errorf("%w: %v", ErrSegmentCancelled, r.terminalErr)
	}
	return ErrSegmentCancelled
}

func (r *SegmentRecorder) manifestLocked(complete, cancelled bool) SegmentManifest {
	segments := make([]SegmentInfo, len(r.segments))
	copy(segments, r.segments)

	return SegmentManifest{
		Version:                      1,
		Format:                       r.format,
		TargetSegmentDurationSeconds: r.duration.Seconds(),
		FramesPerSegment:             r.framesPerSegment,
		TotalFrames:                  r.completedFrames,
		TotalSamples:                 r.completedFrames * uint64(r.format.ChannelCount),
		Segments:                     segments,
		Complete:                     complete,
		Cancelled:                    cancelled,
	}
}

// writeManifestLocked atomically replaces the manifest file: the new content
// is written and synced to a temporary file in the same directory, then
// renamed over the old manifest.
func (r *SegmentRecorder) writeManifestLocked(complete, cancelled bool) error {
	manifest := r.manifestLocked(complete, cancelled)

	tmp, err := os.CreateTemp(r.dir, ".manifest-*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary manifest: %w", err)
	}
	tmpName := tmp.Name()

	cleanup := func() {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
	}

	// CreateTemp uses 0600; make the manifest readable like the segments.
	if err := tmp.Chmod(0o644); err != nil {
		cleanup()
		return fmt.Errorf("set manifest permissions: %w", err)
	}

	encoder := json.NewEncoder(tmp)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(&manifest); err != nil {
		cleanup()
		return fmt.Errorf("encode manifest: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		cleanup()
		return fmt.Errorf("sync manifest: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("close manifest: %w", err)
	}
	if err := os.Rename(tmpName, filepath.Join(r.dir, r.manifest)); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("replace manifest: %w", err)
	}

	return nil
}

// framesForDuration converts a duration to a whole number of frames, rounding
// down. It reports ErrSegmentInvalidDuration for non-positive durations,
// ErrSegmentDurationTooSmall if the result is zero frames, and
// ErrSegmentDurationTooLong on overflow.
func framesForDuration(duration time.Duration, frameRate uint32) (uint64, error) {
	if duration <= 0 {
		return 0, ErrSegmentInvalidDuration
	}

	seconds := uint64(duration / time.Second)
	remainder := uint64(duration % time.Second)
	rate := uint64(frameRate)

	if seconds > math.MaxUint64/rate {
		return 0, ErrSegmentDurationTooLong
	}
	frames := seconds * rate

	// remainder < 1e9 and rate <= 2^32, so this multiplication cannot overflow
	// uint64, but the addition can.
	fractional := (remainder * rate) / uint64(time.Second)
	if frames > math.MaxUint64-fractional {
		return 0, ErrSegmentDurationTooLong
	}
	frames += fractional

	if frames == 0 {
		return 0, ErrSegmentDurationTooSmall
	}
	return frames, nil
}

// maxSegmentDataBytes returns the largest 'data' chunk size that can be
// represented inside a RIFF/WAVE file with the given format.
func maxSegmentDataBytes(format SegmentFormat) uint64 {
	formatData := NewFormatChunkData(
		format.ChannelCount, format.FrameRate, format.SampleType,
	)

	// RIFF ckSize covers: "WAVE" (4) + format chunk (8 + body) + optional
	// fact chunk (12) + data chunk header (8) + data body (and its pad byte).
	overhead := uint64(4 + 8 + formatData.ChunkSize() + 8)
	if formatData.FormatCode != FormatCodePCM {
		overhead += 12
	}

	// Reserve one extra byte for the data chunk's optional pad byte.
	max := uint64(math.MaxUint32) - overhead
	if max > 0 {
		max--
	}

	// The recorder slices segment data with int indexes, so the segment size
	// must also be representable as an int on the current platform. This only
	// matters on 32-bit platforms, where int is smaller than uint32.
	if maxInt := uint64(^uint(0) >> 1); max > maxInt {
		max = maxInt
	}
	return max
}
