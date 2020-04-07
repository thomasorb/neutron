import numpy as np
import multiprocessing
import logging
import time
import re
import traceback

import mido

from . import config
from . import ccore



class Core(object):

    def __init__(self):

        self.manager = multiprocessing.Manager()
        self.p = multiprocessing.RawArray('d', np.zeros(4, dtype=float))
        self.p[0] = config.TEMPO
        self.p[1] = config.MEASURE_BEATS
        self.p[2] = config.MEASURE_NOTE
        self.p[3] = config.OCTAVE

        self.retrack = re.compile(r't')
        self.reint = re.compile(r'\d+')
        
        self.tracks = list()
        for ich in range(config.CHANNEL_NB):
            self.tracks.append(self.manager.list())

        self.midi_server_process = multiprocessing.Process(
            name='midiserver',
            target=MidiServer, 
            args=(self.tracks, self.p))

        self.midi_server_process.start()
        
    def append(self, msg, track=0):
        print('> append: ', msg)
        self.tracks[track].append(msg)

    def replace(self, msg, track=0):
        self.append(msg, track=track)
        while len(self.tracks[track]) > 1:
            self.tracks[track].pop(0)

    def empty(self, track):
        while len(self.tracks[track]) > 0:
            self.tracks[track].pop(0)

    def __lshift__(self, msg):
        try:
            if isinstance(msg, int):
                self.empty(msg)
                return
        
            newmsg = ''
            track = 0
            for im in msg.split():
                if 't' in im:
                    track = int(self.reint.findall(im)[0])
                else:
                    newmsg += ' ' + im
        except Exception as e:
            print(e)
            return
        
        self.append(newmsg, track)

    def enum(self):
        for i in range(len(self.tracks)):
            print('>', i, self.tracks[i])

    def __del__(self):
        print('terminating server')
        try:
            self.midi_server_process.terminate()
            self.midi_server_process.join()
        except Exception as e:
            print(e)
        
        

class MidiServer(object):

    NOTE_DIVISIONS = 2**6
    SLEEP_FACTOR = 1/1000.
    durations = {
        'x':100000,
        'oo':4,
        'o':2,
        'p':0.5,
        'pp':0.25,
        'ppp':1/8}
    accidents = 'd', 'b'
    notes = np.array([0, 2, 4, 5, 7, 9, 11])

    def __del__(self):
        self.outport.panic()
    
    def __init__(self, tracks, p):
        self.tracks = tracks
        self.p = p
        # all modes defined in one line !!
        self.modes = [list(np.roll(self.notes, i)) for i in range(7)]
        self.mode = self.modes[config.MODE]
        
        logging.info('>> MIDI OUTPUTS:\n   {}'.format('\n   '.join(mido.get_output_names())))
        logging.info('>> QUARK MIDI OUTPUT: {}'.format(config.MIDI_OUT))
        self.outport = mido.open_output(config.MIDI_OUT)

        tempo = self.p[0]
        measure = (self.p[1], self.p[2])
        self.octave = self.p[3]
    
        self.t_note =  60 / config.TEMPO # note (noire) duration (s)
        self.t_step = self.t_note / self.NOTE_DIVISIONS # step duration (s)
        self.t_bar = measure[0] / measure[1] * 4 * self.t_note # bar duration (s)
        self.bar_divisions = int(self.NOTE_DIVISIONS * measure[0] / measure[1] * 4)

        self.redur = re.compile(r'[opx]+')
        self.repit = re.compile(r'[\dbd?]+')
        self.renote = re.compile(r'[\d?]+')
        
        
        timing_stats = list()
        timer = time.time()
        step_index = 0

        start_time = time.time()
        
        next_msgs = list()
        for itrack in range(len(self.tracks)):
            next_msgs.append((self.read_last_bar(itrack), itrack))

        msgps = list()
        for i in range(len(self.tracks)):
            msgps.append(list())
        
        while True:
            bar_index = step_index // self.bar_divisions
            
            # read new msgs one step before next bar
            if not (step_index + 1)% self.bar_divisions:
                next_msgs = list()
                for itrack in range(len(self.tracks)):
                    #print('track {} bar {}'.format(itrack, bar_index))
                    # read last bar
                    try:
                        next_msgs.append((self.read_last_bar(itrack), itrack))
                        if 'kill' in next_msgs[-1][0]:
                            for imsgp, ievent in msgps[itrack]:
                                ievent.set()
                                imsgp.join()                                
                            
                        if 'destroy' in next_msgs[-1][0]:
                            # this track is read and destroyed immediately
                            self.tracks[itrack].pop(-1)
                        else:
                            # list is rolled
                            if len(self.tracks[itrack]) > 1:
                                self.tracks[itrack].insert(0, self.tracks[itrack].pop(-1))

                    except Exception as e:
                        print("error reading last bar: {}".format(traceback.print_exc(5)))
                        if len(self.tracks[itrack]) > 0:
                            self.tracks[itrack].pop(-1) # erroneous bar is removed
                        next_msgs.append((msgs[itrack], itrack))

                # clean message processes
                for imsgp_track in msgps:
                    for imsgp, ievent in imsgp_track:
                        if not imsgp.is_alive():
                            imsgp.join()
                            del imsgp


            # first step of a bar
            if not step_index % self.bar_divisions:
                # load msgs for the whole bar
                msgs = next_msgs
                print(msgs)

            for i in range(len(msgs)):
                if str(step_index % self.bar_divisions) in msgs[i][0]:
                    stop_event = multiprocessing.Event()
                    msgp = multiprocessing.Process(
                        name='msg', 
                        target=sendmessage,
                        args=(msgs[i][0][str(step_index % self.bar_divisions)],
                              self.outport, self.t_step, stop_event))
                    msgp.start()
                    msgps[msgs[i][1]].append((msgp, stop_event))

            # step synchronization (must be at the end of the loop)
            step_index += 1
            
            while time.time() - (start_time + step_index * self.t_step) <= self.t_step - (self.t_step * self.SLEEP_FACTOR / 2.):
                time.sleep(self.t_step * self.SLEEP_FACTOR)
            timing_stats.append(time.time() - (start_time + step_index * self.t_step))
            
        print(np.mean(timing_stats))
        print(np.std(timing_stats))

    def read_last_bar(self, track):
        
        def acc(s):
            if 'b' in s:
                return -1
            if 'd' in s:
                return +1
            return 0

        def conv(note, mode, octave):
            return mode[(note - 1) % len(mode)] + (note - 1) // len(mode) * 12 + octave * 12 - 1

        # last bar is read
        instrument = config.INSTRUMENT
        octave = int(self.octave)

        if len(self.tracks[track]) == 0: return dict()

        msgs_dict = dict()
        step = 0
        for im in self.tracks[track][-1].split():
            velocity = config.VELOCITY

            if '!' in im:
                msgs_dict['kill'] = True
                msgs_dict['destroy'] = True
                break
                
            if ':' in im:
                instrument = int(self.renote.findall(im)[0])
                continue

            if '#' in im:
                octave = int(self.renote.findall(im)[0])
                continue

            dur = self.redur.findall(im)
            pit = self.repit.findall(im)
            pitch = list()
            for inum in pit:
                ipitch = self.renote.findall(inum)[0]
                if '?' in ipitch:
                    #np.random.seed()
                    ipitch = self.mode[np.random.randint(len(self.mode))] + np.random.randint(8) * 12
                    print(ipitch)
                else:
                    ipitch = int(ipitch) - 1
                    ipitch = conv(ipitch, self.mode, octave) + acc(inum)
                pitch.append(ipitch)
                    
                         
                # pitch = self.mode[ipit[1] % len(self.mode) + ipit[1] // len(self.mode)]
                # + acc(ipit[0])
                # + octave * 12 for ipit in zip(pit, pitch)

            
            duration = 1
            if len(dur) > 0:
                duration = self.durations[dur[0]]
            
            imsg = [pitch, duration, velocity, instrument]
            print('>>>>>>>>>>>>>>>>', imsg)
            # durations are converted and a message dict is returned
            msgs_dict[str(step)] = [imsg[0], imsg[1] * self.t_note, imsg[2], imsg[3]]
            if dur[0] == 'x':
                msgs_dict['destroy'] = True
                
            step += int(imsg[1] * self.NOTE_DIVISIONS) # note duration in steps

        return msgs_dict
        


def sendmessage(msg, outport, t_step, stop_event):
    msg = msg
    outport = outport
    stime = time.time()
    notes = list()
    for inote in msg[0]:
        print(inote)
        if '?' == inote:
            print('pouet==============')
            np.random.seed()
            notes.append(np.random.randint(0, 127))
            print(notes)
        else:
            notes.append(inote)
            
    for inote in notes:
        outport.send(mido.Message('note_on', note=inote, velocity=msg[2],
                                       channel=msg[3]))

    while time.time() - stime < msg[1] - 8 * t_step and not stop_event.is_set():
        time.sleep(config.SLEEPTIME)

    for inote in notes:
        outport.send(mido.Message('note_off', note=inote, channel=msg[3]))


