import os
from typing import Iterable, Any
from midiutil import MIDIFile


def rle(items: Iterable[Any]) -> Iterable[Any]:
    iterator = iter(items)
    current = next(iterator)
    count = 1

    try:
        while True:
            next_val = next(iterator)
            if next_val == current:
                count += 1
            else:
                yield current, count
                current = next_val
                count = 1
    except StopIteration:
        yield current, count
        return


def main():
    file = 'out.txt'
    path = os.path.join('input', file)
    root, _ = os.path.splitext(file)

    with open(path) as f:
        lines = f.readlines()

    tracks = list(zip(
        *(
            (
                int(note.strip())
                for note
                in line.split('\t')
            )
            for line
            in lines
        )
    ))

    channel = 0
    time = 0  # In beats
    duration = 1  # In beats
    tempo = 70  # In BPM
    volume = 127  # 0-127, as per the MIDI standard

    for a, b in zip(tracks, tracks[1:]):
        len_a = sum(count for _, count in rle(a))
        len_b = sum(count for _, count in rle(b))

        print(len_a, len_b)
        assert len_a == len_b

    # tracks = [tracks[0]]

    midi = MIDIFile(len(tracks))
    for i, track in enumerate(tracks):
        midi.addTempo(i, time, tempo)

        for j, (pitch, count) in enumerate(rle(track)):
            print('j', i, repr(j), repr(pitch), count)

            # pitch 0 means silence
            midi.addNote(i, channel, pitch, time + j, duration * count, volume if pitch > 0 else 0)

    midi_path = os.path.join('output', f'{root}.midi')
    print(f'Writing {midi_path}')

    with open(midi_path, 'wb') as output_file:
        midi.writeFile(output_file)

    # ogg_path = os.path.join('output', f'{root}.ogg')
    # print(f'Writing {ogg_path}')
    # os.system('fluidsynth '
    #           '-nli '
    #           '-r 48000 '
    #           '-o synth.cpu-cores=4 '
    #           '-T oga '
    #           f'-F {ogg_path} '
    #           f'/usr/share/soundfonts/FluidR3_GM.sf2 {midi_path}')


if __name__ == '__main__':
    main()
