// segment-writer demonstrates how to record a long-running, frame-aligned
// sample stream into a sequence of wave files using wave.SegmentRecorder.
//
// Unlike the stream-writer example (which produces a single file), this
// program rotates files on a fixed schedule and writes a manifest that can be
// used to reassemble the original stream. This is useful for long recordings
// where losing a single file should not invalidate the whole session.
package main

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/jonchammer/audio-io/core"
	"github.com/jonchammer/audio-io/wave"
)

func main() {

	const (
		outputDir     = "segments"
		duration      = 5 * time.Second
		segmentLength = time.Second
		sampleRate    = 44100
		channelCount  = 2
		sineFrequency = 440.0 // The note 'A'
		sineGain      = 0.25  // Amplitude multiplier
		bufferFrames  = 300   // Deliberately not a multiple of the segment size
	)

	// 1. Pick the format used for every segment.
	format := wave.SegmentFormat{
		SampleType:   wave.SampleTypeInt16,
		FrameRate:    uint32(sampleRate),
		ChannelCount: channelCount,
	}

	// 2. Create the recorder. This creates 'outputDir' if needed and writes an
	// initial manifest.
	recorder, err := wave.NewSegmentRecorder(outputDir, format, segmentLength)
	if err != nil {
		failF(err)
	}

	// 3. Generate and write audio in small, frame-aligned blocks. The recorder
	// handles opening, rotating, and finalizing segment files.
	totalFrames := int(duration.Seconds() * sampleRate)
	samplingInterval := 1.0 / float64(sampleRate)
	var x float64
	for written := 0; written < totalFrames; {
		frames := bufferFrames
		if remaining := totalFrames - written; remaining < frames {
			frames = remaining
		}

		// Build one interleaved stereo block. Each frame contains one sample
		// per channel.
		samples := make([]float64, frames*channelCount)
		for frame := 0; frame < frames; frame++ {
			value := sineGain * math.Sin(2*math.Pi*sineFrequency*x)
			x += samplingInterval
			for channel := 0; channel < channelCount; channel++ {
				samples[frame*channelCount+channel] = value
			}
		}

		if err := recorder.WriteInt16(core.QuantizeToInt16(samples)); err != nil {
			failF(err)
		}
		written += frames
	}

	// 4. Finish finalizes the last segment and marks the manifest complete.
	// (Placing it in a defer is also a good pattern in real applications.)
	if err := recorder.Finish(); err != nil {
		failF(err)
	}

	manifest := recorder.Manifest()
	fmt.Printf(
		"Recorded %d frames in %d segments (%.1f s per segment) under %q\n",
		manifest.TotalFrames,
		len(manifest.Segments),
		segmentLength.Seconds(),
		outputDir,
	)
}

func failF(err error) {
	_, _ = fmt.Fprintln(os.Stderr, err)
	os.Exit(-1)
}
