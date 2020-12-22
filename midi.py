from mido import MidiFile, MidiTrack, Message, MetaMessage, second2tick
from collections import namedtuple
import numpy as np
import cv2

Encoded = namedtuple('Encoded', 'data key_signature time_signature bpm')
custom_ticks_per_second = 16


def get_metadata_track(tempo, encoded):
    meta_track = MidiTrack()
    meta_track.append(encoded.time_signature)
    meta_track.append(encoded.key_signature)
    meta_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    return meta_track


def to_midi(encoded, midi_path):
    # Tempo is not
    tempo = 500000
    ticks_per_second = second2tick(1, encoded.bpm, tempo)

    mid = MidiFile(type=1, ticks_per_beat=encoded.bpm)
    mid.tracks.append(get_metadata_track(tempo, encoded))

    tracks = list(map(lambda x: MidiTrack(), range(128)))
    prev_column = np.zeros(128, np.int8)
    recent_velocity_changes = np.zeros(len(tracks), int)
    for t, column in enumerate(encoded.data.T):
        difference = column - prev_column
        for note, velocity_change in enumerate(difference):
            if velocity_change == 0:
                continue
            custom_ticks_since_last_change = t - recent_velocity_changes[note]
            midi_ticks_since_last_change = round(
                custom_ticks_since_last_change / custom_ticks_per_second * ticks_per_second)

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
    return Encoded(data, key_signature, time_signature, mid.ticks_per_beat)


if __name__ == '__main__':
    # encoded = from_midi("data/unfin.midi")
    # print("Midi -> Encoded")
    # to_midi(encoded, 'data/unfin_result.midi')
    # print("Encoded -> Midi")

    encoded = from_midi("data/zeppelin.mid")
    print("Midi -> Encoded")
    to_midi(encoded, 'data/zeppelin_result.midi')
    print("Encoded -> Midi")
