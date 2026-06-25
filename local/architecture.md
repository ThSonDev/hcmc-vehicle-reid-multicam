# Local pipeline architecture

Plain-language reference for how the local pipeline works. It explains the logic, not
every line of code; variable and function names are given only where they help you find
the matching code.

## Cameras

Three fixed cameras film the same stretch of traffic from different points. The pipeline
treats cam1 as the "source" and tries to recognise the same vehicles again at cam2 and
cam3.

| Camera | Stream resolution | Ground-truth resolution | View                  |
|--------|-------------------|-------------------------|-----------------------|
| cam1   | 640p              | 1920 x 1080             | front of the vehicles |
| cam2   | 640p              | 1920 x 1080             | back of the vehicles  |
| cam3   | 640p              | 1440 x 1080             | front of the vehicles |

The videos the pipeline actually reads are downscaled to 640p and live in
`data/cam1_640.mp4`, `data/cam2_640.mp4`, `data/cam3_640.mp4`. The ground-truth labels
were drawn on the original full-resolution footage, so they use the larger resolutions in
the table above. The labels are stored as MOT-format text files at the repo root:
`gt_cam1.txt`, `gt_cam2.txt`, `gt_cam3.txt`. The gap between stream resolution and label
resolution is handled later, at evaluation time, not by the producer.

### Road layout and what it means for matching

- **cam1 and cam2 are on the same one-way road.** Traffic flows cam1 then cam2 and there
  is no entrance between them, so every vehicle seen at cam2 must have already passed
  cam1. There are no brand-new vehicles at cam2. Because cam1 sees the front and cam2 sees
  the back, the match has to work across opposite views of the same vehicle.
- **cam3 sits where another road joins.** Vehicles can enter from that side road, so some
  vehicles at cam3 were never seen by cam1. cam3 must therefore be able to say "this is a
  new vehicle" instead of forcing a match. cam3 sees the front again, like cam1.

This is why the two downstream cameras behave differently: cam2 always expects to find a
match, while cam3 is allowed to report new vehicles.

## Producer

The producer (`producer.py`) plays the three video files back as if they were three live
cameras and pushes the frames into Kafka. Everything downstream reads from there.

### One producer, three camera threads

There is a single Kafka `Producer` object, shared by three worker threads, one per camera
(`stream_worker`, started once per entry in `config.VIDEO_SOURCES`). All three publish to
the same topic, `video_reid_stream`, and tell the cameras apart by using the camera id
(`cam1`, `cam2`, `cam3`) as the Kafka message key. Consumers filter on that key.

All three threads are given the same start time (`global_start`, taken once before any
thread starts). They use this shared clock for pacing, which keeps the three streams
roughly time-aligned with each other.

### What one worker does, frame by frame

For its own video file, each worker loops:

1. **Read the next frame.** If the video ends, behaviour depends on replay mode (below):
   either jump back to the start and keep going, or stop.
2. **Wait so playback is real time.** It works out when this frame is "due"
   (frame number divided by the video's real FPS) and compares that to how long the run
   has actually been going. If the frame is early, it sleeps the difference. This makes the
   stream behave like a real camera instead of dumping frames as fast as the disk allows,
   and it is the natural throttle on the whole pipeline.
3. **Skip frames to hit the target rate.** The source video may be a higher FPS than
   wanted, so the worker keeps only every Nth frame (`frame_skip_interval`, derived from
   the video FPS and `TARGET_FPS = 30`) and drops the rest.
4. **Encode the frame as JPEG** (`JPEG_QUALITY = 70`). `TARGET_WIDTH` is `None`, so the
   frame is not resized — the videos are already 640p. This is why the saved results stay
   in 640 coordinates and get scaled up only at evaluation time.
5. **Send it to Kafka.** The JPEG bytes are the message value. A small piece of metadata,
   `{"timestamp", "frame_idx"}`, is attached as a Kafka header named `meta`, not put inside
   the image payload. Consumers trust this header timestamp as the wall-clock time of the
   frame and only fall back to their own clock if they cannot read it.

After sending, the worker calls `producer.poll(0)` so Kafka can run delivery callbacks
without blocking. Every five seconds it logs a heartbeat with the real send rate and the
running total of frames sent.

### Sending without blocking

`producer.produce(...)` does not wait for the broker; it just queues the frame. Delivery
results come back later through the `delivery_report` callback, which only logs failures.
If the outbound queue is full, `produce` raises `BufferError`; the worker catches it, polls
briefly to let the queue drain, and moves on. The Kafka config backs this up with a large
buffer (`queue.buffering.max.messages = 50000`), a short batching delay
(`linger.ms = 10`), and `lz4` compression, since JPEG frames are large.

### Replay mode

By default the producer loops each video forever, which is fine for watching the live demo
but inflates counts if you try to grade a run. Setting the environment variable
`REID_REPLAY=0` makes each worker play its video exactly once and then stop. This is the
clean single pass needed for ground-truth evaluation, and it is what `run.py --once` turns
on for you.

### Shutdown

A `SIGINT` (Ctrl-C) sets a shared `exit_event`, which tells every worker to leave its loop.
The main thread also exits once all workers have finished on their own (the single-pass
case). On the way out it calls `producer.flush(...)` so any frames still queued are
actually delivered before the process ends.

### Why the producer starts last

The consumers read from the newest offset (`auto.offset.reset=latest`), so any frame sent
before they have subscribed is lost. For that reason `run.py` always starts the producer
after the consumers are up.
