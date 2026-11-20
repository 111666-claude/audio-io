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
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// ------------------------------------------------------------------------- //
// Helpers
// ------------------------------------------------------------------------- //

// openSegmentReaders opens every segment listed in 'manifest' in order.
func openSegmentReaders(t *testing.T, dir string, manifest SegmentManifest) []*Reader {
	t.Helper()

	files := make([]*os.File, 0, len(manifest.Segments))
	readers := make([]*Reader, 0, len(manifest.Segments))
	for _, segment := range manifest.Segments {
		file, err := os.Open(filepath.Join(dir, segment.FileName))
		require.NoError(t, err)
		files = append(files, file)
		readers = append(readers, NewReader(file))
	}
	t.Cleanup(func() {
		for _, file := range files {
			_ = file.Close()
		}
	})
	return readers
}

// readAllFrames reads all frames from a list of readers as raw little-endian
// bytes, verifying that each segment's header matches 'format'.
func readAllFrames(t *testing.T, readers []*Reader, format SegmentFormat) []byte {
	t.Helper()

	var result []byte
	for _, reader := range readers {
		header, err := reader.Header()
		require.NoError(t, err)
		require.NoError(t, header.Validate())
		require.Equal(t, format.FrameRate, header.FrameRate())
		require.Equal(t, format.ChannelCount, header.ChannelCount())

		sampleType, err := header.SampleType()
		require.NoError(t, err)
		require.Equal(t, format.SampleType, sampleType)

		data := make([]byte, header.DataBytes)
		read, err := io.ReadFull(reader.dataReader, data)
		require.NoError(t, err)
		require.Equal(t, len(data), read)
		result = append(result, data...)
	}
	return result
}

// encodeFrames encodes interleaved sample values into little-endian bytes.
func encodeFramesInt16(samples []int16) []byte {
	result := make([]byte, len(samples)*2)
	for i, sample := range samples {
		binary.LittleEndian.PutUint16(result[i*2:], uint16(sample))
	}
	return result
}

func readManifestFile(t *testing.T, path string) SegmentManifest {
	t.Helper()

	data, err := os.ReadFile(path)
	require.NoError(t, err)

	var manifest SegmentManifest
	require.NoError(t, json.Unmarshal(data, &manifest))
	return manifest
}

func listFiles(t *testing.T, dir string) []string {
	t.Helper()

	entries, err := os.ReadDir(dir)
	require.NoError(t, err)
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	sort.Strings(names)
	return names
}

// ------------------------------------------------------------------------- //
// Construction / validation
// ------------------------------------------------------------------------- //

func TestNewSegmentRecorder_Normal(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{
		SampleType:   SampleTypeInt16,
		FrameRate:    44100,
		ChannelCount: 2,
	}

	r, err := NewSegmentRecorder(dir, format, 600*time.Second)
	require.NoError(t, err)
	require.Equal(t, format, r.Format())
	require.Equal(t, dir, r.Directory())
	require.Equal(t, uint64(44100*600), r.FramesPerSegment())

	// The manifest is written eagerly so the directory is self-describing.
	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Equal(t, 1, manifest.Version)
	require.Equal(t, format, manifest.Format)
	require.False(t, manifest.Complete)
	require.Empty(t, manifest.Segments)
	require.Equal(t, 600.0, manifest.TargetSegmentDurationSeconds)
}

func TestNewSegmentRecorder_ZeroDuration(t *testing.T) {
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 44100, ChannelCount: 1}

	_, err := NewSegmentRecorder(t.TempDir(), format, 0)
	require.ErrorIs(t, err, ErrSegmentInvalidDuration)

	_, err = NewSegmentRecorder(t.TempDir(), format, -time.Second)
	require.ErrorIs(t, err, ErrSegmentInvalidDuration)
}

func TestNewSegmentRecorder_DurationShorterThanFrame(t *testing.T) {
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 44100, ChannelCount: 1}

	_, err := NewSegmentRecorder(t.TempDir(), format, time.Nanosecond)
	require.ErrorIs(t, err, ErrSegmentDurationTooSmall)
}

func TestNewSegmentRecorder_DurationRoundsDown(t *testing.T) {
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 1000, ChannelCount: 1}

	// 1.5 frames rounds down to a single frame.
	r, err := NewSegmentRecorder(t.TempDir(), format, 1500*time.Microsecond)
	require.NoError(t, err)
	require.Equal(t, uint64(1), r.FramesPerSegment())
}

func TestNewSegmentRecorder_DurationTooLong(t *testing.T) {
	format := SegmentFormat{SampleType: SampleTypeInt32, FrameRate: 192000, ChannelCount: 2}

	// Roughly 15.6 hours of 32-bit stereo is more than a RIFF data chunk can
	// describe. Note that the *total* recording length is not limited.
	_, err := NewSegmentRecorder(t.TempDir(), format, 100*time.Hour)
	require.ErrorIs(t, err, ErrSegmentDurationTooLong)
}

func TestNewSegmentRecorder_InvalidFormat(t *testing.T) {
	dir := t.TempDir()

	_, err := NewSegmentRecorder(dir, SegmentFormat{
		SampleType: SampleType(-1), FrameRate: 44100, ChannelCount: 1,
	}, time.Second)
	require.ErrorIs(t, err, ErrSegmentInvalidFormat)

	_, err = NewSegmentRecorder(dir, SegmentFormat{
		SampleType: SampleTypeInt16, FrameRate: 0, ChannelCount: 1,
	}, time.Second)
	require.ErrorIs(t, err, ErrSegmentInvalidFormat)

	_, err = NewSegmentRecorder(dir, SegmentFormat{
		SampleType: SampleTypeInt16, FrameRate: 44100, ChannelCount: 0,
	}, time.Second)
	require.ErrorIs(t, err, ErrSegmentInvalidFormat)
}

func TestNewSegmentRecorder_Options(t *testing.T) {
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	_, err := NewSegmentRecorder(t.TempDir(), format, time.Second,
		WithManifestFileName(""))
	require.Error(t, err)

	_, err = NewSegmentRecorder(t.TempDir(), format, time.Second,
		WithSegmentFileNamePrefix(""))
	require.Error(t, err)

	// Verifies that a nil context is rejected. The interface makes it possible
	// to pass nil, so this is intentional.
	var nilContext context.Context
	_, err = NewSegmentRecorder(t.TempDir(), format, time.Second, WithContext(nilContext))
	require.Error(t, err)

	// A directory path must be rejected.
	_, err = NewSegmentRecorder("", format, time.Second)
	require.Error(t, err)

	// Names must not escape the output directory.
	_, err = NewSegmentRecorder(t.TempDir(), format, time.Second,
		WithManifestFileName("../manifest.json"))
	require.Error(t, err)

	_, err = NewSegmentRecorder(t.TempDir(), format, time.Second,
		WithSegmentFileNamePrefix("nested/segment-"))
	require.Error(t, err)

	// The manifest must not collide with generated segment names.
	_, err = NewSegmentRecorder(t.TempDir(), format, time.Second,
		WithManifestFileName("segment-notes.wav"))
	require.Error(t, err)

	// Valid custom names are honored.
	dir := t.TempDir()
	r, err := NewSegmentRecorder(dir, format, time.Second,
		WithManifestFileName("index.json"),
		WithSegmentFileNamePrefix("part_"))
	require.NoError(t, err)
	require.NoError(t, r.WriteInt16([]int16{1}))
	require.NoError(t, r.Finish())

	require.Equal(t, []string{"index.json", "part_000001.wav"}, listFiles(t, dir))
	manifest := readManifestFile(t, filepath.Join(dir, "index.json"))
	require.Equal(t, "part_000001.wav", manifest.Segments[0].FileName)
}

func TestNewSegmentRecorder_RefusesExistingRecording(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	_, err := NewSegmentRecorder(dir, format, time.Second)
	require.NoError(t, err)

	// Creating a second recorder in the same directory would overwrite the
	// first recording's manifest, so it must be rejected.
	_, err = NewSegmentRecorder(dir, format, time.Second)
	require.Error(t, err)
	require.Contains(t, err.Error(), "manifest")
}

func TestNewSegmentRecorder_ExistingSegmentIsNotOverwritten(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	// Simulate a previous recording occupying the first segment name. Note
	// that the recorder refuses to start when a manifest already exists, so
	// only a stray segment file is present here.
	existing := filepath.Join(dir, "segment-000001.wav")
	require.NoError(t, os.WriteFile(existing, []byte("keep me"), 0o644))

	r, err := NewSegmentRecorder(dir, format, time.Second)
	require.NoError(t, err)

	err = r.WriteInt16([]int16{1})
	require.Error(t, err)
	require.Contains(t, err.Error(), "segment-000001.wav")

	// The pre-existing file must be untouched.
	data, err := os.ReadFile(existing)
	require.NoError(t, err)
	require.Equal(t, []byte("keep me"), data)
}

// ------------------------------------------------------------------------- //
// Core: per-sample concatenation across segment boundaries
// ------------------------------------------------------------------------- //

func TestSegmentRecorder_ConcatenationInt16Mono(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	// 10 frames per segment; 3.5 segments worth of input, so the last segment
	// is short.
	const framesPerSegment = 10
	const totalFrames = 35
	r, err := NewSegmentRecorder(dir, format, time.Second*framesPerSegment/8000)
	require.NoError(t, err)
	require.Equal(t, uint64(framesPerSegment), r.FramesPerSegment())

	input := make([]int16, totalFrames)
	for i := range input {
		input[i] = int16(i*7 - 100)
	}

	// Write in irregular chunks to exercise carry-over buffering.
	for offset, chunk := 0, 1; offset < len(input); chunk = (chunk*3)%7 + 1 {
		end := offset + chunk
		if end > len(input) {
			end = len(input)
		}
		require.NoError(t, r.WriteInt16(input[offset:end]))
		offset = end
	}
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.True(t, manifest.Complete)
	require.Len(t, manifest.Segments, 4)
	require.Equal(t, uint64(totalFrames), manifest.TotalFrames)
	require.Equal(t, uint64(totalFrames), manifest.TotalSamples)
	require.Equal(t, []uint64{10, 10, 10, 5}, frameCounts(manifest.Segments))
	require.Equal(t, []uint64{0, 10, 20, 30}, startFrames(manifest.Segments))

	// Each segment must match the corresponding slice of input exactly.
	readers := openSegmentReaders(t, dir, manifest)
	expected := encodeFramesInt16(input)
	actual := readAllFrames(t, readers, format)
	require.Equal(t, expected, actual)

	// Verify the manifest's per-segment byte accounting matches the files.
	for i, segment := range manifest.Segments {
		header, err := readers[i].Header()
		require.NoError(t, err)
		require.Equal(t, uint32(segment.DataBytes), header.DataBytes)
		require.Equal(t, uint32(segment.FrameCount), header.FrameCount())
		require.Equal(t, time.Duration(segment.FrameCount)*time.Second/8000, header.PlayTime())
	}
}

func frameCounts(segments []SegmentInfo) []uint64 {
	result := make([]uint64, len(segments))
	for i, segment := range segments {
		result[i] = segment.FrameCount
	}
	return result
}

func startFrames(segments []SegmentInfo) []uint64 {
	result := make([]uint64, len(segments))
	for i, segment := range segments {
		result[i] = segment.StartFrame
	}
	return result
}

func TestSegmentRecorder_ExactMultipleOfSegment(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, 2*time.Second/8000*5)
	require.NoError(t, err) // 10 frames per segment
	require.Equal(t, uint64(10), r.FramesPerSegment())

	input := make([]int16, 30)
	for i := range input {
		input[i] = int16(i)
	}
	require.NoError(t, r.WriteInt16(input))
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Len(t, manifest.Segments, 3)
	require.Equal(t, []uint64{10, 10, 10}, frameCounts(manifest.Segments))

	readers := openSegmentReaders(t, dir, manifest)
	require.Equal(t, encodeFramesInt16(input), readAllFrames(t, readers, format))
}

// TestSegmentRecorder_PulseAcrossBoundary is the multi-channel fixed-impulse
// check: a short pulse is placed so that it straddles the boundary between two
// segments. Each sample in the pulse has a distinct per-channel value and a
// unique pattern, so any dropped, duplicated, or shifted sample at the cut
// would be detected. The test also cross-checks the WAV data-chunk length.
func TestSegmentRecorder_PulseAcrossBoundary(t *testing.T) {
	const (
		channels         = 3
		frameRate        = 8000
		framesPerSegment = 8
		totalFrames      = 23 // boundary at frame 8 and 16
		// The pulse starts 3 frames before the first cut and runs 4 frames
		// past it, so it lands on both sides of the boundary.
		pulseStart = framesPerSegment - 3
		pulseLen   = 7
	)

	dir := t.TempDir()
	format := SegmentFormat{
		SampleType:   SampleTypeInt16,
		FrameRate:    frameRate,
		ChannelCount: channels,
	}
	r, err := NewSegmentRecorder(dir, format, time.Second*framesPerSegment/frameRate)
	require.NoError(t, err)
	require.Equal(t, uint64(framesPerSegment), r.FramesPerSegment())

	// Build an interleaved stream of zeros with a deterministic pulse. For
	// frame f (relative to the pulse start) and channel c, the value is
	// distinct: 1000 + 10*f + c.
	input := make([]int16, totalFrames*channels)
	for f := 0; f < pulseLen; f++ {
		for c := 0; c < channels; c++ {
			input[(pulseStart+f)*channels+c] = int16(1000 + 10*f + c)
		}
	}

	// Deliver the stream in blocks that deliberately split frames across
	// writes (5 samples at a time, not a multiple of the 3-channel block).
	for offset := 0; offset < len(input); offset += 5 {
		end := offset + 5
		if end > len(input) {
			end = len(input)
		}
		require.NoError(t, r.WriteInt16(input[offset:end]))
	}
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.True(t, manifest.Complete)
	require.Len(t, manifest.Segments, 3)
	require.Equal(t, []uint64{8, 8, 7}, frameCounts(manifest.Segments))

	// The pulse must be entirely contained in the first two segments: frames
	// 5..11, i.e. the last three frames of segment 1 and the first four of
	// segment 2.
	readers := openSegmentReaders(t, dir, manifest)
	segments := make([][]byte, len(readers))
	for i, reader := range readers {
		header, err := reader.Header()
		require.NoError(t, err)
		require.NoError(t, header.Validate())
		require.Equal(t, format.FrameRate*uint32(format.BlockAlign()), header.ByteRate())
		require.Equal(t, uint16(format.BlockAlign()), header.FormatData.BlockAlign)

		// The WAV data chunk length must be exactly frames * blockAlign.
		require.Equal(t, uint32(manifest.Segments[i].FrameCount)*uint32(format.BlockAlign()), header.DataBytes)
		segments[i] = make([]byte, header.DataBytes)
		_, err = io.ReadFull(reader.dataReader, segments[i])
		require.NoError(t, err)
	}

	// Reconstruct the stream sample-by-sample and compare against input.
	reconstructed := make([]int16, 0, len(input))
	for _, data := range segments {
		for i := 0; i+1 < len(data); i += 2 {
			reconstructed = append(reconstructed, int16(binary.LittleEndian.Uint16(data[i:])))
		}
	}
	require.Equal(t, len(input), len(reconstructed))
	require.Equal(t, input, reconstructed)

	// Explicitly assert the pulse survives the cut intact, channel by channel.
	for f := 0; f < pulseLen; f++ {
		for c := 0; c < channels; c++ {
			globalSample := (pulseStart+f)*channels + c
			require.Equal(t, int16(1000+10*f+c), reconstructed[globalSample],
				"frame %d channel %d", pulseStart+f, c)
		}
	}

	// Sanity check: the cut falls inside the pulse, not between pulses.
	require.Less(t, pulseStart, framesPerSegment)
	require.Greater(t, pulseStart+pulseLen, framesPerSegment)
}

func TestSegmentRecorder_OddSizedSegmentPadding(t *testing.T) {
	// uint8 mono with an odd frame count per segment produces an odd-sized
	// 'data' chunk. The wave spec requires a pad byte in that case; it must be
	// written (so RIFF sizes are correct) but must not appear as audio data.
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeUint8, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second*5/8000) // 5 frames
	require.NoError(t, err)
	input := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	require.NoError(t, r.WriteUint8(input))
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Equal(t, []uint64{5, 5, 1}, frameCounts(manifest.Segments))

	for _, segment := range manifest.Segments {
		data, err := os.ReadFile(filepath.Join(dir, segment.FileName))
		require.NoError(t, err)
		require.Zero(t, len(data)%2, "wave files must have an even total size")

		// The RIFF size must cover everything after the 8-byte RIFF header.
		require.Equal(t, uint32(len(data)-8), binary.LittleEndian.Uint32(data[4:8]))

		// The data chunk size must exclude the pad byte.
		dataIndex := bytes.Index(data, []byte("data"))
		require.NotEqual(t, -1, dataIndex)
		require.Equal(t, uint32(segment.DataBytes), binary.LittleEndian.Uint32(data[dataIndex+4:]))
		require.Equal(t, int(segment.DataBytes), len(data)-(dataIndex+8)-(int(segment.DataBytes)&1))
	}

	readers := openSegmentReaders(t, dir, manifest)
	require.Equal(t, input, readAllFrames(t, readers, format))
}

// TestSegmentRecorder_StreamingRoundTrip exercises the recorder and the
// existing streaming Reader APIs end to end: samples are written in
// irregular blocks (including the original stream-writer pattern of a
// non-frame-aligned final block), then each segment is read back in small
// blocks with ReadInt16 and compared sample-for-sample.
// TestSegmentRecorder_ConcatenationProperty checks the core invariant over
// many deterministic pseudo-random combinations of channel count, sample
// type, segment length, and (deliberately irregular, non-frame-aligned) write
// sizes: concatenating the segments must reproduce the encoded input exactly,
// with no sample dropped or duplicated at any cut.
func TestSegmentRecorder_ConcatenationProperty(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5E6D3E17))
	formats := []SegmentFormat{
		{SampleType: SampleTypeUint8, FrameRate: 8000, ChannelCount: 1},
		{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 2},
		{SampleType: SampleTypeInt16, FrameRate: 44100, ChannelCount: 5},
		{SampleType: SampleTypeInt24, FrameRate: 48000, ChannelCount: 3},
		{SampleType: SampleTypeInt32, FrameRate: 16000, ChannelCount: 2},
		{SampleType: SampleTypeFloat32, FrameRate: 22050, ChannelCount: 4},
		{SampleType: SampleTypeFloat64, FrameRate: 96000, ChannelCount: 1},
	}

	for iteration := 0; iteration < 40; iteration++ {
		format := formats[rng.Intn(len(formats))]
		blockAlign := format.BlockAlign()

		framesPerSegment := 1 + rng.Intn(20)

		dir := t.TempDir()
		// Use a per-frame duration rounded *up* to whole nanoseconds so that
		// the requested length is never short of the intended frame count.
		frameNanos := (time.Second + time.Duration(format.FrameRate) - 1) /
			time.Duration(format.FrameRate)
		duration := time.Duration(framesPerSegment) * frameNanos

		r, err := NewSegmentRecorder(dir, format, duration)
		require.NoError(t, err)

		// The recorder rounds the duration down to whole frames; use its
		// actual value rather than the requested one.
		framesPerSegment = int(r.FramesPerSegment())
		totalFrames := rng.Intn(5*framesPerSegment + 1)

		input := make([]byte, totalFrames*blockAlign)
		_, err = rng.Read(input)
		require.NoError(t, err)

		// Write in random chunks that usually do not align to frame edges.
		for offset := 0; offset < len(input); {
			size := 1 + rng.Intn(3*blockAlign+1)
			if remaining := len(input) - offset; size > remaining {
				size = remaining
			}
			require.NoError(t, r.WriteRaw(input[offset:offset+size]))
			offset += size
		}
		require.NoError(t, r.Finish())

		manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
		require.True(t, manifest.Complete)
		require.Equal(t, uint64(totalFrames), manifest.TotalFrames)

		// Exactly one segment per full segment's worth of frames, plus one for
		// the remainder (if any).
		expectedSegments := totalFrames / framesPerSegment
		if totalFrames%framesPerSegment != 0 {
			expectedSegments++
		}
		require.Len(t, manifest.Segments, expectedSegments)

		var expectedStart uint64
		for _, segment := range manifest.Segments {
			require.Equal(t, expectedStart, segment.StartFrame)
			require.Greater(t, segment.FrameCount, uint64(0))
			require.LessOrEqual(t, segment.FrameCount, uint64(framesPerSegment))
			require.Equal(t, segment.FrameCount*uint64(blockAlign), segment.DataBytes)
			expectedStart += segment.FrameCount
		}
		require.Equal(t, uint64(totalFrames), expectedStart)

		readers := openSegmentReaders(t, dir, manifest)
		actual := readAllFrames(t, readers, format)
		if len(input) == 0 {
			require.Empty(t, actual)
		} else {
			require.Equal(t, input, actual,
				"iteration %d, format %+v, %d frames of %d per segment",
				iteration, format, totalFrames, framesPerSegment)
		}
	}
}

func TestSegmentRecorder_StreamingRoundTrip(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 44100, ChannelCount: 2}

	const (
		framesPerSegment = 257 // deliberately not a power of two
		bufferFrames     = 100 // smaller than a segment, not a divisor
	)
	r, err := NewSegmentRecorder(dir, format, time.Duration(framesPerSegment)*time.Second/44100)
	require.NoError(t, err)

	// A ramp with per-channel offsets makes any shifted sample detectable.
	totalFrames := 3*framesPerSegment + 37
	input := make([]int16, totalFrames*2)
	for f := 0; f < totalFrames; f++ {
		input[f*2] = int16(f % 30000)
		input[f*2+1] = int16(-(f % 30000))
	}

	for offset := 0; offset < len(input); offset += bufferFrames * 2 {
		end := offset + bufferFrames*2
		if end > len(input) {
			end = len(input)
		}
		require.NoError(t, r.WriteInt16(input[offset:end]))
	}
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.True(t, manifest.Complete)
	require.Len(t, manifest.Segments, 4)

	var output []int16
	for i, reader := range openSegmentReaders(t, dir, manifest) {
		header, err := reader.Header()
		require.NoError(t, err)
		require.Equal(t, uint16(2), header.ChannelCount())
		require.Equal(t, uint32(manifest.Segments[i].SampleCount), header.SampleCount())

		buffer := make([]int16, bufferFrames*2)
		for {
			n, err := reader.ReadInt16(buffer)
			output = append(output, buffer[:n]...)
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				break
			}
			require.NoError(t, err)
		}
	}

	require.Equal(t, input, output)
}

func TestSegmentRecorder_AllSampleTypes(t *testing.T) {
	testCases := []struct {
		name   string
		format SegmentFormat
		write  func(t *testing.T, r *SegmentRecorder, frames int)
	}{
		{
			name:   "Uint8",
			format: SegmentFormat{SampleType: SampleTypeUint8, FrameRate: 8000, ChannelCount: 2},
			write: func(t *testing.T, r *SegmentRecorder, frames int) {
				data := make([]uint8, frames*2)
				for i := range data {
					data[i] = uint8(i)
				}
				require.NoError(t, r.WriteUint8(data))
			},
		},
		{
			name:   "Int16",
			format: SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 2},
			write: func(t *testing.T, r *SegmentRecorder, frames int) {
				data := make([]int16, frames*2)
				for i := range data {
					data[i] = int16(i * 13)
				}
				require.NoError(t, r.WriteInt16(data))
			},
		},
		{
			name:   "Int24",
			format: SegmentFormat{SampleType: SampleTypeInt24, FrameRate: 8000, ChannelCount: 2},
			write: func(t *testing.T, r *SegmentRecorder, frames int) {
				data := make([]int32, frames*2)
				for i := range data {
					data[i] = int32(i*1000 - 8388608)
				}
				require.NoError(t, r.WriteInt24(data))
			},
		},
		{
			name:   "Int32",
			format: SegmentFormat{SampleType: SampleTypeInt32, FrameRate: 8000, ChannelCount: 2},
			write: func(t *testing.T, r *SegmentRecorder, frames int) {
				data := make([]int32, frames*2)
				for i := range data {
					data[i] = int32(i * 100000)
				}
				require.NoError(t, r.WriteInt32(data))
			},
		},
		{
			name:   "Float32",
			format: SegmentFormat{SampleType: SampleTypeFloat32, FrameRate: 8000, ChannelCount: 2},
			write: func(t *testing.T, r *SegmentRecorder, frames int) {
				data := make([]float32, frames*2)
				for i := range data {
					data[i] = float32(i) / 100
				}
				require.NoError(t, r.WriteFloat32(data))
			},
		},
		{
			name:   "Float64",
			format: SegmentFormat{SampleType: SampleTypeFloat64, FrameRate: 8000, ChannelCount: 2},
			write: func(t *testing.T, r *SegmentRecorder, frames int) {
				data := make([]float64, frames*2)
				for i := range data {
					data[i] = float64(i) / 100
				}
				require.NoError(t, r.WriteFloat64(data))
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			const framesPerSegment = 5
			const totalFrames = 13

			r, err := NewSegmentRecorder(dir, tc.format, time.Second*framesPerSegment/8000)
			require.NoError(t, err)
			tc.write(t, r, totalFrames)
			require.NoError(t, r.Finish())

			manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
			require.Len(t, manifest.Segments, 3)
			require.Equal(t, uint64(totalFrames), manifest.TotalFrames)

			readers := openSegmentReaders(t, dir, manifest)
			data := readAllFrames(t, readers, tc.format)
			require.Equal(t, int(manifest.TotalFrames)*tc.format.BlockAlign(), len(data))

			// Round-trip through the typed reader to prove the segments are
			// valid files for their declared format.
			verifyRoundTrip(t, dir, manifest, tc.format, totalFrames)
		})
	}
}

func verifyRoundTrip(t *testing.T, dir string, manifest SegmentManifest, format SegmentFormat, totalFrames int) {
	t.Helper()

	readers := openSegmentReaders(t, dir, manifest)
	var result []float64
	for _, reader := range readers {
		header, err := reader.Header()
		require.NoError(t, err)

		switch format.SampleType {
		case SampleTypeUint8:
			data := make([]uint8, header.SampleCount())
			n, _ := reader.ReadUint8(data)
			for _, v := range data[:n] {
				result = append(result, float64(v))
			}
		case SampleTypeInt16:
			data := make([]int16, header.SampleCount())
			n, _ := reader.ReadInt16(data)
			for _, v := range data[:n] {
				result = append(result, float64(v))
			}
		case SampleTypeInt24:
			data := make([]int32, header.SampleCount())
			n, _ := reader.ReadInt24(data)
			for _, v := range data[:n] {
				result = append(result, float64(v))
			}
		case SampleTypeInt32:
			data := make([]int32, header.SampleCount())
			n, _ := reader.ReadInt32(data)
			for _, v := range data[:n] {
				result = append(result, float64(v))
			}
		case SampleTypeFloat32:
			data := make([]float32, header.SampleCount())
			n, _ := reader.ReadFloat32(data)
			for _, v := range data[:n] {
				result = append(result, float64(v))
			}
		case SampleTypeFloat64:
			data := make([]float64, header.SampleCount())
			n, _ := reader.ReadFloat64(data)
			result = append(result, data[:n]...)
		}
	}
	require.Equal(t, totalFrames*int(format.ChannelCount), len(result))
}

func TestSegmentRecorder_TypedWriteRejectsWrongType(t *testing.T) {
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}
	r, err := NewSegmentRecorder(t.TempDir(), format, time.Second)
	require.NoError(t, err)

	require.ErrorIs(t, r.WriteUint8([]uint8{1}), ErrWriterExpectedUint8)
	require.ErrorIs(t, r.WriteInt24([]int32{1}), ErrWriterExpectedInt24)
	require.ErrorIs(t, r.WriteInt32([]int32{1}), ErrWriterExpectedInt32)
	require.ErrorIs(t, r.WriteFloat32([]float32{1}), ErrWriterExpectedFloat32)
	require.ErrorIs(t, r.WriteFloat64([]float64{1}), ErrWriterExpectedFloat64)

	// None of the failed calls should have opened a segment.
	require.Equal(t, []string{"manifest.json"}, listFiles(t, r.Directory()))
}

func TestSegmentRecorder_WriteBlockFormatMismatch(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 2}
	r, err := NewSegmentRecorder(dir, format, time.Second*4/8000)
	require.NoError(t, err)

	// A block in the expected format is accepted.
	require.NoError(t, r.WriteBlock(format, encodeFramesInt16([]int16{1, 2})))

	// Changing the channel count mid-recording is rejected...
	err = r.WriteBlock(SegmentFormat{
		SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1,
	}, encodeFramesInt16([]int16{9}))
	require.ErrorIs(t, err, ErrSegmentFormatMismatch)

	// ...as is changing the sample type or frame rate.
	err = r.WriteBlock(SegmentFormat{
		SampleType: SampleTypeFloat32, FrameRate: 8000, ChannelCount: 2,
	}, make([]byte, 8))
	require.ErrorIs(t, err, ErrSegmentFormatMismatch)

	err = r.WriteBlock(SegmentFormat{
		SampleType: SampleTypeInt16, FrameRate: 44100, ChannelCount: 2,
	}, make([]byte, 4))
	require.ErrorIs(t, err, ErrSegmentFormatMismatch)

	// An invalid format is reported as such.
	err = r.WriteBlock(SegmentFormat{}, make([]byte, 0))
	require.ErrorIs(t, err, ErrSegmentInvalidFormat)

	// The recorder stays usable and the rejected data was not written.
	require.NoError(t, r.WriteBlock(format, encodeFramesInt16([]int16{3, 4})))
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Equal(t, uint64(2), manifest.TotalFrames)
	require.Equal(t, encodeFramesInt16([]int16{1, 2, 3, 4}),
		readAllFrames(t, openSegmentReaders(t, dir, manifest), format))
}

func TestSegmentRecorder_WriteBlockRejectsAmbiguousFrameSize(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 2}
	r, err := NewSegmentRecorder(dir, format, time.Second)
	require.NoError(t, err)

	// Buffer a partial frame, then attempt a block too short to complete it.
	// The block is rejected because its frame boundary is ambiguous.
	require.NoError(t, r.WriteRaw([]byte{1, 0}))
	err = r.WriteBlock(format, []byte{2})
	require.ErrorIs(t, err, ErrSegmentFormatMismatch)
	require.Contains(t, err.Error(), "partial frame")

	// An empty block is always a no-op, even with a partial frame buffered.
	require.NoError(t, r.WriteBlock(format, nil))

	// A block that does complete the buffered frame is accepted (and may
	// contain further whole frames).
	require.NoError(t, r.WriteBlock(format, []byte{2, 0, 3, 0, 4, 0}))
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Equal(t, uint64(2), manifest.TotalFrames)
	require.Equal(t, encodeFramesInt16([]int16{1, 2, 3, 4}),
		readAllFrames(t, openSegmentReaders(t, dir, manifest), format))
}

func TestSegmentRecorder_WriteBlockMatchesTypedAPI(t *testing.T) {
	// The same input must produce identical files whether it is supplied as
	// encoded bytes via WriteBlock or as typed samples via WriteInt16.
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 3}
	input := make([]int16, 7*3)
	for i := range input {
		input[i] = int16(i*11 - 50)
	}
	encoded := encodeFramesInt16(input)

	type result struct {
		manifest SegmentManifest
		data     []byte
	}
	record := func(write func(r *SegmentRecorder)) result {
		dir := t.TempDir()
		r, err := NewSegmentRecorder(dir, format, time.Second*3/8000)
		require.NoError(t, err)
		write(r)
		require.NoError(t, r.Finish())
		manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
		return result{manifest: manifest, data: readAllFrames(t, openSegmentReaders(t, dir, manifest), format)}
	}

	typed := record(func(r *SegmentRecorder) { require.NoError(t, r.WriteInt16(input)) })
	raw := record(func(r *SegmentRecorder) { require.NoError(t, r.WriteBlock(format, encoded)) })

	require.Equal(t, typed.manifest, raw.manifest)
	require.Equal(t, typed.data, raw.data)
}

// ------------------------------------------------------------------------- //
// Partial frames and short writes
// ------------------------------------------------------------------------- //

func TestSegmentRecorder_CarryAcrossWriteCalls(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 4}

	r, err := NewSegmentRecorder(dir, format, time.Second)
	require.NoError(t, err)

	// 4 channels * 2 bytes = 8-byte frames. Feed one byte at a time.
	input := make([]byte, 8*3)
	for i := range input {
		input[i] = byte(i)
	}
	for _, b := range input {
		require.NoError(t, r.WriteRaw([]byte{b}))
	}
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Equal(t, uint64(3), manifest.TotalFrames)
	require.Equal(t, input, readAllFrames(t, openSegmentReaders(t, dir, manifest), format))
}

func TestSegmentRecorder_PartialFrameAtFinish(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 2}

	r, err := NewSegmentRecorder(dir, format, time.Second)
	require.NoError(t, err)

	// 2 complete frames plus 3 stray bytes (one and a half frames).
	require.NoError(t, r.WriteRaw([]byte{1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6}))

	err = r.Finish()
	require.ErrorIs(t, err, ErrSegmentPartialFrame)

	// All complete frames are preserved and the segment is valid.
	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.False(t, manifest.Complete)
	require.Equal(t, uint64(2), manifest.TotalFrames)
	require.Equal(t, []byte{1, 0, 2, 0, 3, 0, 4, 0}, readAllFrames(t, openSegmentReaders(t, dir, manifest), format))

	// Finish is idempotent and keeps reporting the same outcome.
	require.ErrorIs(t, r.Finish(), ErrSegmentPartialFrame)
	require.ErrorIs(t, r.WriteRaw([]byte{7, 0}), ErrSegmentPartialFrame)
}

func TestSegmentRecorder_WriteAfterFinish(t *testing.T) {
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}
	r, err := NewSegmentRecorder(t.TempDir(), format, time.Second)
	require.NoError(t, err)

	require.NoError(t, r.Finish())
	require.NoError(t, r.Finish())
	require.ErrorIs(t, r.WriteInt16([]int16{1}), ErrSegmentFinished)
	require.ErrorIs(t, r.WriteRaw([]byte{1, 0}), ErrSegmentFinished)
}

// shortFile wraps an *os.File and returns a short write on a chosen call.
type shortFile struct {
	*os.File
	writeCalls int
	failAt     int
}

func (f *shortFile) Write(p []byte) (int, error) {
	f.writeCalls++
	if f.writeCalls == f.failAt {
		// Claim to have written some (but not all) of the bytes, with no error.
		n := len(p) / 2
		if n == 0 {
			n = 1
		}
		_, err := f.File.Write(p[:n])
		return n, err
	}
	return f.File.Write(p)
}

func TestSegmentRecorder_ShortWrite(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second*4/8000) // 4 frames
	require.NoError(t, err)

	// Fail the second segment's first data write.
	openCount := 0
	r.openFile = func(path string) (segmentFile, error) {
		openCount++
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if err != nil {
			return nil, err
		}
		if openCount == 2 {
			return &shortFile{File: file, failAt: 3}, nil // 1st write = header
		}
		return file, nil
	}

	input := make([]int16, 12)
	for i := range input {
		input[i] = int16(i)
	}
	err = r.WriteInt16(input)
	require.Error(t, err)
	require.ErrorIs(t, err, io.ErrShortWrite)

	// The first segment stays valid and present; the failed segment is gone.
	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Len(t, manifest.Segments, 1)
	require.False(t, manifest.Complete)
	require.Equal(t, []uint64{4}, frameCounts(manifest.Segments))
	require.Equal(t, []string{"manifest.json", "segment-000001.wav"}, listFiles(t, dir))

	readers := openSegmentReaders(t, dir, manifest)
	require.Equal(t, encodeFramesInt16(input[:4]), readAllFrames(t, readers, format))

	// Once failed, the recorder rejects further writes.
	require.ErrorIs(t, r.WriteInt16([]int16{1}), io.ErrShortWrite)
}

func TestSegmentRecorder_CommitShortWrite(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second*4/8000)
	require.NoError(t, err)

	// Fail the final Flush (header rewrite) of the first segment.
	openCount := 0
	r.openFile = func(path string) (segmentFile, error) {
		openCount++
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if err != nil {
			return nil, err
		}
		if openCount == 1 {
			return &shortFile{File: file, failAt: 3}, nil // header, data, header rewrite
		}
		return file, nil
	}

	err = r.WriteInt16(make([]int16, 4))
	require.ErrorIs(t, err, io.ErrShortWrite)

	// An incomplete segment must not appear in the manifest.
	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Empty(t, manifest.Segments)
	require.Equal(t, []string{"manifest.json"}, listFiles(t, dir))
}

// ------------------------------------------------------------------------- //
// Disk failures
// ------------------------------------------------------------------------- //

func TestSegmentRecorder_OpenFailure(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second*2/8000) // 2 frames
	require.NoError(t, err)

	diskErr := errors.New("disk full")
	openCount := 0
	r.openFile = func(path string) (segmentFile, error) {
		openCount++
		if openCount == 2 {
			return nil, diskErr
		}
		return os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	}

	err = r.WriteInt16(make([]int16, 6))
	require.ErrorIs(t, err, diskErr)

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Len(t, manifest.Segments, 1)
	require.Equal(t, uint64(2), manifest.TotalFrames)
	require.Equal(t, []string{"manifest.json", "segment-000001.wav"}, listFiles(t, dir))
}

func TestSegmentRecorder_SyncFailure(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second*2/8000)
	require.NoError(t, err)

	syncErr := errors.New("sync failed")
	openCount := 0
	r.openFile = func(path string) (segmentFile, error) {
		openCount++
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if err != nil {
			return nil, err
		}
		if openCount == 2 {
			return &syncFailFile{File: file, err: syncErr}, nil
		}
		return file, nil
	}

	err = r.WriteInt16(make([]int16, 6))
	require.ErrorIs(t, err, syncErr)

	// The unsynced segment is discarded; the first remains valid.
	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Len(t, manifest.Segments, 1)
	require.Equal(t, []string{"manifest.json", "segment-000001.wav"}, listFiles(t, dir))
}

// TestSegmentRecorder_FramesWrittenAfterFailure verifies that frames belonging
// to a segment that could not be finalized are not reported as written.
func TestSegmentRecorder_FramesWrittenAfterFailure(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second*2/8000) // 2 frames
	require.NoError(t, err)

	syncErr := errors.New("sync failed")
	r.openFile = func(path string) (segmentFile, error) {
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if err != nil {
			return nil, err
		}
		return &syncFailFile{File: file, err: syncErr}, nil
	}

	err = r.WriteInt16([]int16{1, 2})
	require.ErrorIs(t, err, syncErr)
	require.Equal(t, uint64(0), r.FramesWritten())
	require.Equal(t, uint64(0), r.Manifest().TotalFrames)
}

type syncFailFile struct {
	*os.File
	err error
}

func (f *syncFailFile) Sync() error {
	return f.err
}

func TestSegmentRecorder_WriterCreationFailure(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second)
	require.NoError(t, err)

	// A failed open must not leave a file behind or affect the manifest.
	openErr := errors.New("cannot create file")
	r.openFile = func(path string) (segmentFile, error) {
		// Simulate a create followed by a failure; the recorder must not
		// touch the path itself (the opener owns cleanup).
		require.NoError(t, os.WriteFile(path, nil, 0o644))
		return nil, openErr
	}

	require.ErrorIs(t, r.WriteInt16([]int16{1}), openErr)
	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Empty(t, manifest.Segments)
}

func TestSegmentRecorder_ManifestWriteFailure(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second*2/8000)
	require.NoError(t, err)

	// Replace the manifest with a non-empty directory so the atomic rename
	// cannot succeed, while the segment file itself can still be created.
	manifestPath := filepath.Join(dir, "manifest.json")
	require.NoError(t, os.Remove(manifestPath))
	require.NoError(t, os.Mkdir(manifestPath, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(manifestPath, "blocker"), []byte("x"), 0o644))

	// The first rotation triggers a manifest update, which fails. The segment
	// that triggered it is already complete and remains on disk.
	writeErr := r.WriteInt16(make([]int16, 2))
	require.Error(t, writeErr)
	require.Contains(t, writeErr.Error(), "manifest")

	// The segment was fully written before the manifest update failed, so it
	// remains a valid, completed segment on disk even though the manifest
	// could not be updated to mention it.
	segmentPath := filepath.Join(dir, "segment-000001.wav")
	file, err := os.Open(segmentPath)
	require.NoError(t, err)
	defer func() { _ = file.Close() }()

	reader := NewReader(file)
	header, err := reader.Header()
	require.NoError(t, err)
	require.NoError(t, header.Validate())
	require.Equal(t, uint32(4), header.DataBytes)

	// The failed recorder cannot continue.
	require.ErrorIs(t, r.WriteInt16([]int16{1}), writeErr)
}

// ------------------------------------------------------------------------- //
// Cancellation
// ------------------------------------------------------------------------- //

func TestSegmentRecorder_CancelKeepsCompletedSegments(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second*2/8000) // 2 frames
	require.NoError(t, err)

	// Two full segments plus one frame of a third.
	require.NoError(t, r.WriteInt16([]int16{1, 2, 3, 4, 5}))
	require.Equal(t, uint64(5), r.FramesWritten())

	require.NoError(t, r.Cancel())
	require.ErrorIs(t, r.WriteInt16([]int16{6}), ErrSegmentCancelled)

	// Only the two complete segments survive.
	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.True(t, manifest.Cancelled)
	require.False(t, manifest.Complete)
	require.Len(t, manifest.Segments, 2)
	require.Equal(t, uint64(4), manifest.TotalFrames)
	require.Equal(t, uint64(4), r.FramesWritten())
	require.Equal(t, []string{"manifest.json", "segment-000001.wav", "segment-000002.wav"}, listFiles(t, dir))

	// Cancel is idempotent, and Finish after Cancel still reports cancellation.
	require.ErrorIs(t, r.Cancel(), ErrSegmentCancelled)
	require.ErrorIs(t, r.Finish(), ErrSegmentCancelled)

	readers := openSegmentReaders(t, dir, manifest)
	expected := encodeFramesInt16([]int16{1, 2, 3, 4})
	require.Equal(t, expected, readAllFrames(t, readers, format))
}

func TestSegmentRecorder_CancelAfterFinishIsNoop(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second)
	require.NoError(t, err)
	require.NoError(t, r.WriteInt16([]int16{1}))
	require.NoError(t, r.Finish())
	require.NoError(t, r.Cancel())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.True(t, manifest.Complete)
	require.False(t, manifest.Cancelled)
	require.Len(t, manifest.Segments, 1)
}

func TestSegmentRecorder_ContextCancellation(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	ctx, cancel := context.WithCancel(context.Background())
	r, err := NewSegmentRecorder(dir, format, time.Second*2/8000, WithContext(ctx))
	require.NoError(t, err)

	require.NoError(t, r.WriteInt16([]int16{1, 2, 3}))
	cancel()

	err = r.WriteInt16([]int16{4, 5})
	require.ErrorIs(t, err, ErrSegmentCancelled)

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.True(t, manifest.Cancelled)
	require.Len(t, manifest.Segments, 1)
	require.Equal(t, []string{"manifest.json", "segment-000001.wav"}, listFiles(t, dir))

	// Finish also observes cancellation.
	require.ErrorIs(t, r.Finish(), ErrSegmentCancelled)
}

func TestSegmentRecorder_ContextCancellationAfterFinish(t *testing.T) {
	// A context cancelled after the recording has already finished or failed
	// must not rewrite the manifest or change the reported outcome.
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	ctx, cancel := context.WithCancel(context.Background())
	r, err := NewSegmentRecorder(dir, format, time.Second, WithContext(ctx))
	require.NoError(t, err)
	require.NoError(t, r.WriteInt16([]int16{1}))
	require.NoError(t, r.Finish())

	cancel()
	require.ErrorIs(t, r.WriteInt16([]int16{2}), ErrSegmentFinished)
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.True(t, manifest.Complete)
	require.False(t, manifest.Cancelled)
	require.Len(t, manifest.Segments, 1)
}

// ------------------------------------------------------------------------- //
// Long recordings
// ------------------------------------------------------------------------- //

// TestSegmentRecorder_LongRecording writes enough frames to produce many
// segments without keeping the entire input (or output) in memory. It checks
// that the segment index, start positions, and ordering remain correct.
func TestSegmentRecorder_LongRecording(t *testing.T) {
	if testing.Short() {
		t.Skip("long recording test skipped in short mode")
	}

	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 2}

	const (
		framesPerSegment = 1000
		segmentCount     = 120 // ~2.5 minutes of audio across 120 files
	)
	r, err := NewSegmentRecorder(dir, format, time.Duration(framesPerSegment)*time.Second/8000)
	require.NoError(t, err)

	var nextValue int16 = 1
	block := make([]int16, framesPerSegment*int(format.ChannelCount))
	for s := 0; s < segmentCount; s++ {
		for i := 0; i < framesPerSegment; i++ {
			for c := 0; c < int(format.ChannelCount); c++ {
				block[i*int(format.ChannelCount)+c] = nextValue
				nextValue++
			}
		}
		require.NoError(t, r.WriteInt16(block))
	}
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.True(t, manifest.Complete)
	require.Len(t, manifest.Segments, segmentCount)

	var expectedStart uint64
	for i, segment := range manifest.Segments {
		require.Equal(t, i+1, segment.Index)
		require.Equal(t, expectedStart, segment.StartFrame)
		require.Equal(t, expectedStart*2, segment.StartSample)
		require.Equal(t, uint64(framesPerSegment), segment.FrameCount)
		require.Equal(t, fmt.Sprintf("segment-%06d.wav", i+1), segment.FileName)
		expectedStart += segment.FrameCount
	}
	require.Equal(t, uint64(segmentCount*framesPerSegment), manifest.TotalFrames)

	// Spot-check the first frame of a few segments to ensure the data was
	// written to the right file in the right order.
	for _, i := range []int{0, 1, 57, segmentCount - 1} {
		reader := openSegmentReaders(t, dir, SegmentManifest{
			Segments: manifest.Segments[i : i+1],
		})[0]
		data := make([]int16, 2)
		n, err := reader.ReadInt16(data)
		require.NoError(t, err)
		require.Equal(t, 2, n)

		firstFrame := uint64(i * framesPerSegment)
		require.Equal(t, int16(firstFrame*2+1), data[0])
		require.Equal(t, int16(firstFrame*2+2), data[1])
	}
}

func TestSegmentRecorder_ManySmallSegments(t *testing.T) {
	// One frame per segment: the maximum possible rotation rate. This checks
	// that segments are never zero-length and that boundaries are exact.
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second/8000)
	require.NoError(t, err)
	require.Equal(t, uint64(1), r.FramesPerSegment())

	input := make([]int16, 100)
	for i := range input {
		input[i] = int16(i)
	}
	require.NoError(t, r.WriteInt16(input))
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Len(t, manifest.Segments, 100)
	for _, segment := range manifest.Segments {
		require.Equal(t, uint64(1), segment.FrameCount)
	}
	require.Equal(t, encodeFramesInt16(input), readAllFrames(t, openSegmentReaders(t, dir, manifest), format))
}

// ------------------------------------------------------------------------- //
// Manifest details
// ------------------------------------------------------------------------- //

func TestSegmentRecorder_ManifestReflectsOnlyCompletedSegments(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second*2/8000) // 2 frames
	require.NoError(t, err)

	require.NoError(t, r.WriteInt16([]int16{1}))
	require.NoError(t, r.WriteInt16([]int16{2}))

	// One segment completed; the in-progress score is empty.
	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Len(t, manifest.Segments, 1)
	require.Equal(t, uint64(2), manifest.TotalFrames)

	snapshot := r.Manifest()
	require.Equal(t, manifest.Segments, snapshot.Segments)
	require.False(t, snapshot.Complete)
}

func TestSegmentRecorder_ManifestJSONIsStable(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeFloat32, FrameRate: 48000, ChannelCount: 6}

	r, err := NewSegmentRecorder(dir, format, 600*time.Second)
	require.NoError(t, err)
	require.NoError(t, r.WriteFloat32(make([]float32, 6)))
	require.NoError(t, r.Finish())

	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	require.NoError(t, err)

	var raw map[string]any
	require.NoError(t, json.Unmarshal(data, &raw))

	// Spot-check the on-disk representation that other tools will consume.
	formatJSON := raw["format"].(map[string]any)
	require.Equal(t, "Float32", formatJSON["sampleType"])
	require.Equal(t, float64(48000), formatJSON["frameRate"])
	require.Equal(t, float64(6), formatJSON["channelCount"])

	segments := raw["segments"].([]any)
	require.Len(t, segments, 1)
	segment := segments[0].(map[string]any)
	require.Equal(t, "segment-000001.wav", segment["fileName"])
	require.Equal(t, float64(0), segment["startFrame"])
	require.Equal(t, float64(0), segment["startSample"])
	require.Equal(t, float64(1), segment["frameCount"])
	require.Equal(t, float64(6), segment["sampleCount"])
	require.Equal(t, float64(24), segment["dataBytes"])
}

// ------------------------------------------------------------------------- //
// Helpers / edge cases
// ------------------------------------------------------------------------- //

func TestSegmentRecorder_WriteEmptyIsNoop(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second)
	require.NoError(t, err)
	require.NoError(t, r.WriteInt16(nil))
	require.NoError(t, r.WriteRaw(nil))
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.True(t, manifest.Complete)
	require.Empty(t, manifest.Segments)
	require.Equal(t, []string{"manifest.json"}, listFiles(t, dir))
}

func TestSegmentRecorder_FinishWithNoData(t *testing.T) {
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1}

	r, err := NewSegmentRecorder(dir, format, time.Second)
	require.NoError(t, err)
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.True(t, manifest.Complete)
	require.Empty(t, manifest.Segments)
}

func TestSegmentRecorder_WriteRawEnforcesFixedFormat(t *testing.T) {
	// WriteRaw is format-agnostic at the byte level; it is the caller's
	// responsibility to supply matching bytes. This test documents that a
	// trailing partial frame is buffered and completed.
	dir := t.TempDir()
	format := SegmentFormat{SampleType: SampleTypeInt32, FrameRate: 8000, ChannelCount: 3}
	r, err := NewSegmentRecorder(dir, format, time.Second)
	require.NoError(t, err)

	frame := make([]byte, 12)
	for i := range frame {
		frame[i] = byte(i)
	}
	require.NoError(t, r.WriteRaw(frame[:5]))
	require.NoError(t, r.WriteRaw(frame[5:]))
	require.NoError(t, r.Finish())

	manifest := readManifestFile(t, filepath.Join(dir, "manifest.json"))
	require.Equal(t, frame, readAllFrames(t, openSegmentReaders(t, dir, manifest), format))
}

func TestFramesForDuration(t *testing.T) {
	testCases := []struct {
		duration  time.Duration
		frameRate uint32
		expected  uint64
		err       error
	}{
		{time.Second, 44100, 44100, nil},
		{10 * time.Minute, 48000, 28800000, nil},
		{time.Millisecond, 1000, 1, nil},
		{1500 * time.Microsecond, 1000, 1, nil},
		{time.Nanosecond, 44100, 0, ErrSegmentDurationTooSmall},
		{time.Duration(math.MaxInt64), math.MaxUint32, 0, ErrSegmentDurationTooLong},
		{0, 44100, 0, ErrSegmentInvalidDuration},
		{-time.Second, 44100, 0, ErrSegmentInvalidDuration},
	}

	for _, tc := range testCases {
		frames, err := framesForDuration(tc.duration, tc.frameRate)
		if tc.err != nil {
			require.ErrorIs(t, err, tc.err, "%s", tc.duration)
			continue
		}
		require.NoError(t, err, "%s", tc.duration)
		require.Equal(t, tc.expected, frames, "%s", tc.duration)
	}
}

func TestMaxSegmentDataBytes(t *testing.T) {
	// PCM formats have a 16-byte format chunk; float formats use an 18-byte
	// format chunk and add a 12-byte fact chunk.
	pcm := maxSegmentDataBytes(SegmentFormat{
		SampleType: SampleTypeInt16, FrameRate: 8000, ChannelCount: 1,
	})
	float := maxSegmentDataBytes(SegmentFormat{
		SampleType: SampleTypeFloat32, FrameRate: 8000, ChannelCount: 1,
	})

	// On 64-bit platforms the difference is exactly the extra 14 bytes of
	// metadata. On 32-bit platforms both values are clamped to MaxInt32.
	maxInt := uint64(^uint(0) >> 1)
	if pcm < maxInt && float < maxInt {
		require.Equal(t, pcm-14, float)
	}

	// The recorded chunk sizes must fit in a uint32 RIFF field, including the
	// data chunk header and the pad byte.
	require.LessOrEqual(t, pcm+4+8+16+8+1, uint64(math.MaxUint32))

	// The limit must also be addressable with int on this platform, since the
	// recorder slices segment data using int indexes.
	require.LessOrEqual(t, pcm, maxInt)
}
