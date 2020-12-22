from mido import MidiFile, MidiTrack, Message, second2tick, tick2second
from collections import namedtuple
import numpy as np
import cv2


def to_midi(arr, midi_path):
    n_tracks = (arr != 0).sum(axis=0).max()
    bpm = 96
    ticks_per_seconds = second2tick(1, bpm, 500000)

    mid = MidiFile()
    tracks = list(map(lambda x: MidiTrack(), range(n_tracks * 2)))
    prev_column = np.zeros(128, np.int8)
    recent_velocity_changes = np.zeros(len(tracks), np.int8)
    for t, column in enumerate(arr.T):
        column_track_index = 0
        difference = column - prev_column
        for note, velocity_change in enumerate(difference):
            if velocity_change == 0:
                continue
            time_since_last_change = t - recent_velocity_changes[column_track_index]
            time = round(time_since_last_change / 16 * ticks_per_seconds)
            print(time_since_last_change)

            if column[note] == 0:
                message = Message('note_off', note=note, velocity=0, time=time)
            else:
                message = Message('note_on', note=note, velocity=column[note], time=time)

            recent_velocity_changes[column_track_index] = t
            tracks[column_track_index].append(message)
            column_track_index += 1
        prev_column = column

    for track in tracks:
        mid.tracks.append(track)
    mid.save(midi_path)


def from_midi(midi_path, img_output='output/arr.png'):
    Note = namedtuple('Note', 'value time velocity')
    a = MidiFile(midi_path)
    time = 0
    notes = []
    max_t = 0
    for msg in a:
        time += msg.time
        if msg.type == 'note_on' or msg.type == 'note_off':
            t = int(time * 16)
            max_t = t if t > max_t else max_t
            v = msg.velocity if msg.type == 'note_on' else 0
            notes.append(Note(msg.note, t, v))

    arr = np.zeros((128, max_t), np.int8)
    for note in notes:
        arr[note.value, note.time:] = note.velocity

    cv2.imwrite(img_output, arr)
    # print(arr)
    return arr


if __name__ == '__main__':
    arr = from_midi("output/unfin.midi")
    to_midi(arr, 'output/unfin_result.midi')

    # from_midi("output/unfin_result.midi", 'output/unfin_result.png')
    # from_midi("output/unfin.midi", 'output/unfin.png')
