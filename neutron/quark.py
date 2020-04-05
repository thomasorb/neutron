import numpy as np
import multiprocessing

import mido

import logging
import time

from . import config
from . import ccore

class MidiServer(object):

    def __init__(self, ):
        logging.info('>> MIDI OUTPUTS:\n   {}'.format('\n   '.join(mido.get_output_names())))
        logging.info('>> QUARK MIDI OUTPUT: {}'.format(config.MIDI_OUT))
        outport = mido.open_output(config.MIDI_OUT)
        while True:
            note = np.random.randint(50, 110)
            length = np.random.uniform(0.2, 3)
            msg = mido.Message('note_on', note=note, velocity=3, time=length)
            outport.send(msg)

            time.sleep(length)
            msg = mido.Message('note_off', note=note)
            outport.send(msg)



class MidiPlayer(object):

    def __init__(self, out_bufferL, out_bufferR, out_i, notes, p, outlock, data):

        loopcount = 0
        longloop = 100
        timing = 0
        view = ccore.data2view(np.ascontiguousarray(data.astype(np.complex64)))
        
        logging.info('>> MIDI INPUTS:\n   {}'.format('\n   '.join(mido.get_input_names())))
        logging.info('>> MIDI INPUT: {}'.format(config.MIDI_IN))
        inport = mido.open_input(config.MIDI_IN)

        sounds = list()

        release = p[1]
        attack = p[0]
        posx = int(p[2])
        posy = int(p[3])
        
        while True:
            loopcount += 1
            if loopcount > longloop:
                release = p[1]
                attack = p[0]
                for isound in sounds:
                    if not isound[0].is_alive():
                        del isound

            time.sleep(config.SLEEPTIME)

            rawmsgs = [msg for msg in inport.iter_pending()]
            if len(rawmsgs) == 0: continue
            # integrate other messages sent right after the first ones
            time.sleep(config.SLEEPTIME)
            rawmsgs += [msg for msg in inport.iter_pending()] 

            msgs = list()
            msgs.append(rawmsgs[0])
            for msg in rawmsgs[1:]:
                if msg not in msgs:
                    msgs.append(msg)

            # note off are considered first
            msgs = sorted(msgs, key= lambda elem: elem.type == 'note_on')

            for msg in msgs:
                logging.info('MIDI msg: {}'.format(msg))

                if msg.type == 'note_on':
                    timing = time.time()

                    notes[int(msg.note)] = timing
                    
                    
                    sounds.append(
                        (multiprocessing.Process(
                            name='sound', 
                            target=ccore.sound,
                            args=(out_bufferL, out_bufferR, out_i, notes,
                                  msg.note, msg.velocity, msg.channel, outlock,
                                  timing, attack, release,
                                  config.BUFFERSIZE, config.MASTER, config.SLEEPTIME,
                                  view, 'square', config.DIRTY, config.DATATUNE, posx, posy)),
                         msg.note))

                    sounds[-1][0].start()


                else:
                    notes[int(msg.note)] = 0
