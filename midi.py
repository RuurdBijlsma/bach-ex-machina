from mido import MidiFile, MidiTrack, Message, MetaMessage, second2tick
from collections import namedtuple
import numpy as np
import cv2

Encoded = namedtuple('Encoded', 'data key_signature time_signature bpm')
custom_ticks_per_second = 8
# Todo make ticks per second customizable by making midi a class


def get_metadata_track(tempo, encoded):
    meta_track = MidiTrack()
    meta_track.append(encoded.time_signature)
    meta_track.append(encoded.key_signature)
    meta_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    return meta_track


def to_relative_ticks(ticks_per_second, t, prev_t):
    t_since_last_change = t - prev_t
    return round(t_since_last_change / custom_ticks_per_second * ticks_per_second)


# This extracts a continuous note from the given array and returns the corresponding messages
def pop_hold_at(data, start_t, note, prev_t, tps):
    start_velocity = 0
    end_t = start_t
    messages = []

    row_from_t = data[start_t:, note]
    # This `t` starts at 0, so not an absolute value for t
    for t, velocity in enumerate(row_from_t):
        # If it's been processed remove from array
        row_from_t[t] = 0
        if velocity == start_velocity:
            continue
        elif velocity == 0:
            end_t = start_t + t
            time = to_relative_ticks(tps, start_t + t, prev_t)
            messages.append(Message('note_off', note=note, velocity=0, time=time))
            break
        else:
            # Note start or note velocity change within this hold
            time = to_relative_ticks(tps, start_t + t, prev_t)
            prev_t = start_t + t
            start_velocity = velocity
            messages.append(Message('note_on', note=note, velocity=velocity, time=time))

    return messages, end_t


def to_midi(encoded, midi_path):
    tempo = 500000
    tps = second2tick(1, encoded.bpm, tempo)

    mid = MidiFile(type=1, ticks_per_beat=encoded.bpm)
    mid.tracks.append(get_metadata_track(tempo, encoded))
    data = encoded.data.T.copy()
    n_tracks = (data != 0).sum(axis=1).max()

    for track_index in range(0, n_tracks):
        start_search = (0, 0)
        track = MidiTrack()
        while start_search[0] < len(data):
            found_bar = False
            # print(start_search)
            for t in range(start_search[0], len(data)):
                column = data[t]
                prev_column = data[t - 1] if t > 0 else np.zeros(128, np.int8)
                for note in range(start_search[1], len(column)):
                    velocity = column[note]
                    prev_velocity = prev_column[note]
                    if velocity > 0 and prev_velocity == 0:
                        # Found start of a hold (needs to be actual start of that hold bar)
                        # Pass start_search[0] because that's what the time attribute needs to be relative to
                        messages, hold_end_t = pop_hold_at(data, t, note, start_search[0], tps)
                        for message in messages:
                            track.append(message)
                        # Search for a new hold bar starting at this point
                        start_search = (hold_end_t, 0)
                        found_bar = True
                        break
                if found_bar:
                    break
            if not found_bar:
                break
        mid.tracks.append(track)

    mid.save(midi_path)
    print(f"✅ Encoded -> Midi ({midi_path})")


def from_midi(midi_path, img_output='data/arr.png'):
    Note = namedtuple('Note', 'value time velocity')
    mid = MidiFile(midi_path)
    key_signature = None
    time_signature = None
    time = 0
    notes = []
    max_t = 0
    for msg in mid:
        time += msg.time
        if msg.type == 'key_signature':
            key_signature = msg
        elif msg.type == 'time_signature':
            time_signature = msg
        elif msg.type == 'note_on' or msg.type == 'note_off':
            t = int(time * custom_ticks_per_second)
            max_t = t if t > max_t else max_t
            v = msg.velocity if msg.type == 'note_on' else 0
            notes.append(Note(msg.note, t, v))

    data = np.zeros((128, max_t), np.int8)
    for note in notes:
        data[note.value, note.time:] = note.velocity

    cv2.imwrite(img_output, data)
    print(f"✅ Midi ({midi_path}) -> Encoded")
    return Encoded(data, key_signature, time_signature, mid.ticks_per_beat)


if __name__ == '__main__':
    encoded = from_midi("data/unfin.midi")
    to_midi(encoded, 'data/unfin_result.midi')
    # encoded = from_midi("data/zeppelin.mid")
    # to_midi(encoded, 'data/zeppelin_result.midi')
