from mido import MidiFile, MidiTrack, tick2second, second2tick
from collections import namedtuple
import numpy as np
import math
import cv2

Note = namedtuple('Note', 'value time velocity')


def track_to_notes(track):
    time = 0
    notes = []
    for msg in track:
        time += msg.time
        if msg.type == 'note_on' or msg.type == 'note_off':
            notes.append(Note(msg.note, time, msg.velocity if msg.type == 'note_on' else 0))
    return notes


def notes_to_array(tick_duration):
    def x(notes):
        arr = np.zeros((128, tick_duration), np.int8)
        for note in notes:
            arr[note.value, note.time:] = note.velocity

        return arr

    return x


def from_midi(midi_path):
    a = MidiFile(midi_path)
    tempo = get_tempo(a)
    tick_duration = math.ceil(second2tick(a.length, a.ticks_per_beat, tempo))
    notes_data = list(filter(lambda x: x, map(track_to_notes, a.tracks)))
    array_data = list(map(notes_to_array(tick_duration), notes_data))

    for i, arr in enumerate(array_data):
        cv2.imwrite(f"output/img{i}.png", arr)

    print("done")


def get_tempo(midi_file):
    for track in midi_file.tracks:
        for note in track:
            if note.type == "set_tempo":
                return note.tempo
    return 50000


if __name__ == '__main__':
    from_midi("output/unfin.midi")
    # from_midi("output/zeppelin.mid")
    # a = np.zeros((10, 10), np.int8)
    # a[1, 5:] = 30
    # print(a)
