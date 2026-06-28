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

1. **Advance the decoder (`grab`).** `cap.grab()` moves to the next frame without decoding
   or copying it — cheap. If the video ends, behaviour depends on replay mode (below):
   either jump back to the start (re-anchoring the clock, see step 2) and keep going, or stop.
2. **Wait so playback is real time.** It works out when this frame is "due"
   (frame number divided by the video's real FPS) and compares that to how long the run has
   been going, measured on a **monotonic** clock anchored once at start (NTP-safe). If the
   frame is early it sleeps the difference; if pacing has fallen more than `MAX_LAG_SEC`
   (~1s) behind — a stall — it **re-anchors** rather than bursting a backlog of frames into
   Kafka to catch up. This makes the stream behave like a real camera instead of dumping
   frames as fast as the disk allows, and it is the natural throttle on the whole pipeline.
3. **Skip frames to hit the target rate.** The source may run at a higher FPS than wanted,
   so the worker keeps only every Nth frame (`frame_skip_interval`, derived from the video
   FPS and `TARGET_FPS = 30`). Skipped frames were only `grab`bed (step 1), never decoded;
   it `retrieve()`s — the actual decode — *only* for the frames it will send. That split is
   why dropping FPS saves real CPU.
4. **Encode the kept frame as JPEG** (`JPEG_QUALITY = 70`). `TARGET_WIDTH` is `None`, so the
   frame is not resized — the videos are already 640p. This is why the saved results stay
   in 640 coordinates and get scaled up only at evaluation time.
5. **Send it to Kafka.** The JPEG bytes are the message value. A small piece of metadata,
   `{"timestamp", "frame_idx"}`, is attached as a Kafka header named `meta`, not put inside
   the image payload. The timestamp is wall-clock (`time.time()`, for cross-camera travel
   gating); pacing uses the monotonic clock separately. Consumers trust this header
   timestamp and only fall back to their own clock if they cannot read it.

After sending, the worker calls `producer.poll(0)` so Kafka can run delivery callbacks
without blocking. Every five seconds it logs a heartbeat with the real send rate and the
running total of frames sent.

### Sending without blocking

`producer.produce(...)` does not wait for the broker; it just queues the frame. Delivery
results come back later through the `delivery_report` callback, which only logs failures.
If the outbound queue is full, `produce` raises `BufferError`; the worker catches it, polls
briefly to let the queue drain, and moves on. The Kafka config sizes the buffer to ride out
a transient broker/disk stall rather than for throughput: `queue.buffering.max.messages =
2000` capped at 64 MB of memory (`queue.buffering.max.kbytes = 65536`, since JPEG frames are
large), a short batching delay (`linger.ms = 10`), and `lz4` compression. Steady state is
~90 msg/s with a queue depth ≤5, so the heartbeat's `queue` field is the backpressure signal
to watch.

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
after the consumers are up. (Separately, `run.py` pre-creates all Kafka topics before
launching anything, so consumers never subscribe to a missing topic — otherwise cam1's
lazily-created `reid_gallery_stream` wouldn't be seen by cam2/cam3 for ~5 minutes on a cold
broker.)

## Consumers

There is one consumer process per camera (`consumer_cam1.py`, `consumer_cam2.py`,
`consumer_cam3.py`). They all read frames from `video_reid_stream`, but they play very
different roles, set by the road layout described above:

- **cam1 is the source.** It detects and tracks vehicles, turns each one into a feature
  vector ("embedding"), and publishes those embeddings to `reid_gallery_stream`. It never
  matches anything — it only builds the gallery the others search against.
- **cam2 and cam3 are matchers.** They also detect and track, but in addition they listen
  to `reid_gallery_stream`, keep a local copy of cam1's gallery, and try to recognise each
  of their own vehicles in it. When they decide on a match (or, for cam3, decide a vehicle
  is new) they publish a small event to `reid_matches` for the visualizer to display.

So the data flows in one direction: cam1 → `reid_gallery_stream` → cam2/cam3, and both
matchers → `reid_matches` → visualizer.

### What every consumer shares

All three are built from the same pieces, wired up at startup:

- A **YOLO detector** (`weights/cam{1,2}.pt`; cam3 currently reuses cam1's weights) finds
  vehicles in each frame.
- **ByteTrack** (`sv.ByteTrack`) links those detections across frames so each vehicle keeps
  a stable `tracker_id` (a "track") while it is on screen. These per-camera track IDs are
  local — cam2's track 7 has nothing to do with cam1's track 7.
- An **OSNet feature extractor** (`reid_utils.FeatureExtractor`, weights
  `weights/osnet_cam123.pth`) turns a cropped vehicle image into a 512-d embedding. Two
  embeddings are compared with a dot product; because the embeddings are L2-normalised, that
  dot product is cosine similarity in `[-1, 1]`, where higher means "more likely the same
  vehicle".
- A **`MOTResultWriter`** (`reid_utils`) appends every tracked box to
  `results/res_cam*.txt` in MOT format — this is what evaluation later grades.
- Kafka **consumer** and **producer** handles from the `reid_utils` helpers.

They also share the same loop skeleton, which matters for reading the code:

1. **Heartbeat first, on the wall clock.** At the top of every iteration the consumer
   checks whether `HEARTBEAT_SEC` (5s) has passed and, if so, logs an `event=heartbeat`
   with its current FPS and counts. Doing this *before* polling means a consumer that is
   receiving no frames still beats (with `fps=0`) instead of looking crashed — "alive but
   starved" stays distinguishable from "dead".
2. **Poll Kafka briefly** (`consumer.poll`, ~10–20 ms). A `None` result or an error just
   loops again.
3. **Route by topic and key.** Frames on `video_reid_stream` are filtered by message key
   (`cam1`/`cam2`/`cam3`) so each consumer only decodes its own camera. cam2/cam3 also
   receive `reid_gallery_stream` messages, which are handled separately (see below) and
   then `continue`.
4. **Decode the frame and read the timestamp.** The JPEG bytes are decoded with
   `cv2.imdecode`. The authoritative time and frame index come from the Kafka `meta`
   header the producer attached; the consumer falls back to `time.time()` and its own
   counter only if the header can't be read. The header's `frame_idx` is 0-based, so the
   consumer adds 1 to line up with the 1-based GT files when writing results.
5. **Process only every Nth frame.** Each consumer runs detection/tracking on roughly one
   frame in three (cam1 uses `SKIP_FRAMES = 2`, i.e. `frame_count % 3 == 0`; cam2/cam3 use
   `% 3` directly). This is the main CPU lever — the GPU work (YOLO + OSNet) only happens on
   the kept frames.
6. On a processed frame: **detect → `merge_truck_boxes` → track → write results**, then do
   the camera-specific work, then draw the UI and check for `q` to quit.

`merge_truck_boxes` (in `reid_utils`, shared with the Airflow copy) is applied right after
detection on every camera. YOLO tends to split one long vehicle (bus/truck, COCO classes
5/7) into two boxes; this helper stitches two big-vehicle boxes back together when they are
horizontally close (`gap < 50px`) and vertically overlapping (>60% of the shorter box), so
the tracker sees one vehicle instead of two.

A common crop guard runs before any embedding: boxes smaller than `MIN_AREA_THRESHOLD`
(800 px²) are skipped, because tiny far-away crops give unreliable features. Crops are
always clamped to the frame bounds first.

### cam1 — the gallery source

cam1 detects and tracks like everyone else, then publishes embeddings. The key detail is
**throttling**: a vehicle stays in view for many frames, but cam1 does not want to flood the
gallery with a near-identical embedding every frame. It keeps a dictionary
`cam1_last_sent` mapping `track_id → last send time`, and for each tracked vehicle it only
publishes again once more than `SEND_INTERVAL` (0.5s) has passed since its last send. Each
publish to `reid_gallery_stream` carries the track id, the wall-clock timestamp, the
embedding, and a base64-encoded JPEG of the crop (so the downstream cameras and the
visualizer can show cam1's picture of the vehicle). The crop image is what later proves the
match visually.

cam1 batches all valid crops in a frame and calls `extract_batch` once, rather than
extracting one crop at a time — the same batching pattern every consumer uses to keep the
GPU efficient.

### cam2 — frame-by-frame matcher

cam2 keeps a local copy of cam1's gallery in `gallery_db`, keyed by cam1 track id, holding
each vehicle's latest embedding, timestamp, and crop image. Gallery messages only ever
**update** entries here; cam2 never deletes from it.

For its own vehicles, cam2 matches **on every processed frame** (the front-vs-back views
mean the same vehicle can look quite different, so it keeps trying as the vehicle crosses).
The logic:

1. Collect query crops for every current track that isn't already matched (tracks in
   `locked_cam2_map` are skipped) and passes the area guard.
2. Embed them in one batch.
3. Build the candidate gallery: cam1 entries that aren't already taken (not in
   `locked_cam1_ids`) **and** pass the travel-time gate `1.0s < (cam2_ts − cam1_ts) < 10s`.
   That gate encodes the physical fact that a vehicle takes a realistic, bounded time to
   drive from cam1 to cam2 — it both removes impossible matches and shrinks the search.
4. Compute the full similarity matrix (`q_feats · gallery_featsᵀ`) and, for each query
   vehicle, take its best gallery candidate. If that score clears
   `SIMILARITY_THRESHOLD` (0.65) and the cam1 id isn't already locked, it's a match.

**Locking** is what keeps matches one-to-one: on a match cam2 adds the cam1 id to
`locked_cam1_ids` and records `locked_cam2_map[cam2_track] = cam1_id`. After that, the cam1
vehicle can't be claimed by another cam2 track, and the cam2 track stops querying. Each
match is published to `reid_matches` as `{cam_source:"cam2", cam1_id, match_id, score,
timestamp, cam1_b64, match_b64}` and drawn on screen with the matched cam1 id over the box.
Because cam1 and cam2 are on the same one-way road with no entrance between them, cam2
assumes every vehicle eventually has a match and never reports "new".

### cam3 — best-shot matcher on a joining road

cam3 sits where a side road merges, so some of its vehicles were never seen by cam1; it must
be able to say **"new vehicle"** instead of forcing a match. It also sees the front of
vehicles (like cam1), and the longer drive from cam1 means a wider travel-time gate:
`20s < gap < 40s`.

Rather than matching every frame, cam3 waits until a vehicle has **left** and matches once,
using its best image. This is the "best-shot buffering" idea:

1. **Buffer the best shot.** While a vehicle is tracked, cam3 keeps an entry in
   `best_track_buffer[track_id]` holding the largest crop seen so far (`area`), its
   timestamp, and a `missing_count`. Each frame the vehicle is present resets
   `missing_count` to 0; if the current crop is bigger than the stored one, it replaces it.
   Largest box ≈ closest/clearest view, which gives the best embedding.
2. **Detect exit.** Any buffered track *not* seen in the current frame has its
   `missing_count` incremented. Once it exceeds `EXIT_FRAME_THRESHOLD` (20 frames without a
   sighting) the vehicle is considered gone and moved to a `finished_tracks` list.
3. **Match the finished tracks, once.** Their best crops are embedded in one batch and
   compared against the travel-time-filtered, unlocked gallery, exactly like cam2. If the
   best score clears `SIMILARITY_THRESHOLD` (0.5, lower than cam2's because the cam1→cam3
   gap and viewpoint make matching harder) it emits a match event (`is_new=False`) and locks
   the cam1 id. If nothing clears the bar, it emits a **new-vehicle** event (`is_new=True`)
   with no `cam1_id`. Either way the track id goes into `processed_cam3_ids` so flicker (the
   same id reappearing) doesn't reprocess it.

cam3 also prunes its gallery: roughly every 50 inserts it drops entries older than
`MAX_TRAVEL_TIME`, since a cam1 vehicle that old can no longer be a valid cam3 match. cam2
doesn't bother — its gate is much shorter.

The new-vehicle events are what populate the second pane of the visualizer; a later cam3
match for the same source vehicle overrides them.

### Shared consumer modules (`reid_utils.py`)

Beyond `merge_truck_boxes`, two classes carry most of the shared consumer logic:

- **`FeatureExtractor`** wraps the OSNet model. At init it builds the `osnet_x1_0`
  architecture and loads the fine-tuned weights, **dropping the `classifier` head** (only
  the embedding is wanted, not class scores) and tolerating a missing file by warning and
  running with random weights rather than crashing. `extract_batch` converts crops
  BGR→RGB, resizes to 256×128, normalises with ImageNet statistics, runs them through the
  model in one `no_grad` pass, and **L2-normalises** the output so downstream dot products
  are cosine similarities. It takes a `logger=` so its messages land in the calling
  consumer's log instead of opening its own file.
- **`MOTResultWriter`** writes one MOT-format line per tracked box
  (`frame,id,left,top,w,h,conf,…`) to `results/res_cam*.txt`, flushing each line. It can
  rescale boxes from stream resolution to GT resolution, but in this pipeline every camera
  passes `target_width=None`, so results are written in 640 coordinates and the res→GT
  scaling happens later at evaluation time (see the resolution discussion above).

The Kafka helpers (`get_kafka_consumer` / `get_kafka_producer`) and the base64 image
codecs (`encode_image_base64` / `decode_image_base64`) also live here. The consumer config
sets `auto.offset.reset=latest` (read only new frames) and a 10s
`topic.metadata.refresh.interval.ms` as a backstop, so a lazily-created topic is discovered
in ~10s instead of librdkafka's 5-minute default — defence in depth behind `run.py`'s
topic pre-creation.
