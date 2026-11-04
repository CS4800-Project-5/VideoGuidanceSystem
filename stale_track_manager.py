class StaleTrackManager:
    def __init__(self, max_missing_frames=15):
        if max_missing_frames < 1:
            raise ValueError("max_missing_frames must be at least 1")

        self.max_missing_frames = int(max_missing_frames)
        self.active_ids = set()
        self.missing_frames = {}

    def update(self, current_ids):
        """Update observed IDs and return (entered_ids, removed_stale_ids).

        An ID is considered stale when it has been absent for
        max_missing_frames consecutive processed frames.
        """
        current_ids = set(current_ids)
        entered_ids = current_ids - self.active_ids

        # A visible track is active again and its missing counter resets.
        for track_id in current_ids:
            self.active_ids.add(track_id)
            self.missing_frames[track_id] = 0

        removed_stale_ids = []
        for track_id in list(self.active_ids - current_ids):
            self.missing_frames[track_id] = (
                self.missing_frames.get(track_id, 0) + 1
            )

            if self.missing_frames[track_id] >= self.max_missing_frames:
                removed_stale_ids.append(track_id)
                self.active_ids.remove(track_id)
                del self.missing_frames[track_id]

        return entered_ids, removed_stale_ids 