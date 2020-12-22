from mido import MidiFile, MidiTrack, Message, second2tick
from collections import namedtuple
import numpy as np
import cv2


def to_midi(arr, midi_path):
    bpm = 96
    ticks_per_second = second2tick(1, bpm, 500000)

    mid = MidiFile(type=1, ticks_per_beat=bpm)

    tracks = list(map(lambda x: MidiTrack(), range(128)))
    prev_column = np.zeros(128, np.int8)
    recent_velocity_changes = np.zeros(len(tracks), int)
    for t, column in enumerate(arr.T):
        difference = column - prev_column
        for note, velocity_change in enumerate(difference):
            if velocity_change == 0:
                continue
            custom_ticks_since_last_change = t - recent_velocity_changes[note]
            midi_ticks_since_last_change = round(custom_ticks_since_last_change / 16 * ticks_per_second)

            if column[note] == 0:
                message = Message('note_off', note=note, velocity=0, time=midi_ticks_since_last_change)
            else:
                message = Message('note_on', note=note, velocity=column[note], time=midi_ticks_since_last_change)

            recent_velocity_changes[note] = t
            tracks[note].append(message)
        prev_column = column

    for track in tracks:
        mid.tracks.append(track)
    mid.save(midi_path)


def from_midi(midi_path, img_output='data/arr.png'):
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
    return arr


if __name__ == '__main__':
    # Round trip:
    arr = from_midi("data/unfin.midi")
    to_midi(arr, 'data/unfin_result.midi')
