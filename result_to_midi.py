from music21 import *
import os 

circle_of_fifths = {
    "CM": 0,
    "GM": 1,
    "DM": 2,
    "AM": 3,
    "EM": 4,
    "BM": 5,
    "F#M": 6,
    "C#M": 7,
    "FM": -1,
    "BbM": -2,
    "EbM": -3,
    "AbM": -4,
    "DbM": -5,
    "GbM": -6,
    "CbM": -7,
}

duration_to_number = {
    'double_whole': 8,
    'double_whole.': 12, 
    'double_whole_fermata': 8,
    'whole': 4,
    'whole.': 6,
    'whole_fermata': 4,
    'half': 2,
    'half.': 3,
    'half_fermata': 2,
    'half._fermata': 3,
    'quarter': 1,
    'quarter.': 1.5,
    'quarter..': 1.75,
    'quarter_fermata': 1,
    'quarter._fermata': 1.5,
    'quarter.._fermata': 1.75,
    'eighth': 0.5,
    'eighth.': 0.75,
    'eighth..': 0.875,
    'eighth_fermata': 0.5,
    'eighth._fermata': 0.75,
    'sixteenth': 0.25,
    'sixteenth.': 0.375,
    'sixteenth_fermata': 0.25,
    'thirty_second': 0.125,
    'thirty_second.': 0.1875,
    'sixty_fourth': 0.0625,
    'hundred_twenty_eighth': 0.03125,
    # 'quadruple_whole'
    # 'quadruple_whole.'
}

def barline():
    pass

def makeClef(value):
    return clef.clefFromString(value)

def makeGracenote():
    pass

def makeKeySignature(value):
    return key.KeySignature(circle_of_fifths[value])

def makeMultirest():
    pass

def makeNote(value):
    
    note_symbol, duration_symbol = 0, 0
    note_name, note_pitch = 0, 0
    
    try:
        note_symbol = value.split('_', 1)[0]
        duration_symbol = value.split('_', 1)[1]
    except:
        print("makeNote Error")
        
    note_symbol.replace('b', '-')
    
    new_note = note.Note(note_symbol)
    new_note.duration.quarterLength = duration_to_number[duration_symbol]
    
    return new_note

def makeRest(value):
    new_rest = note.Rest()
    new_rest.duration.quarterLength = duration_to_number[value]
    return new_rest

def makeTie():
    pass

def makeTimeSignature(value):
    ts = 0
    if value == "C":
        ts = meter.TimeSignature("4/4")
        ts.symbol = 'common'
    elif value == "C/":
        ts = meter.TimeSignature("2/2")
        ts.symbol = 'cut'
    else:
        ts = meter.TimeSignature(value)
    return ts

def result_to_midi(note_list, score_name):
    
    midi_dir = './midi'
    midi_path = os.path.join(midi_dir, score_name + ".midi")
    test_stream = stream.Stream()
    test_measure = stream.Measure()

    for symbol in note_list:
        symbol_type, symbol_value = 0, 0
        
        try:
            symbol_type = symbol.split('-')[0]
            symbol_value = symbol.split('-')[1]
        except IndexError:
            symbol_type = symbol.split('-')[0]

        if symbol_type == 'barline':
            test_stream.append(test_measure)
            test_measure = stream.Measure()
        elif symbol_type == 'clef':
            test_stream.append(makeClef(symbol_value))
        elif symbol_type == 'gracenote':
            pass
        elif symbol_type == 'keySignature':
            test_stream.append(makeKeySignature(symbol_value))
        elif symbol_type == 'multirest':
            pass
        elif symbol_type == 'note':
            test_measure.append(makeNote(symbol_value))
        elif symbol_type == 'rest':
            test_measure.append(makeRest(symbol_value))
        elif symbol_type == 'tie':
            pass
        elif symbol_type == 'timeSignature':
            test_stream.append(makeTimeSignature(symbol_value))

    test_stream.write('xml', fp=midi_path)
    # test_stream.show()