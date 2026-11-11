"""Regression checks: calibration leakage, persistence, full timeline, cache identity, API shape.
Run from the project root: python -m unittest discover -s tests -v
These use fake visual models; they do NOT measure detection quality.
"""
import base64
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import judge, state_engine as se
from pipeline.classify import flow_windows
from pipeline.recall_policy import META, choose_threshold


def facts(n=16):
    keys = ['person_visible','cashier_at_desk','client_at_desk','outsider_behind_counter',
            'collector_present','cash_on_desk','cash_in_hands','collection_bag_visible','safe_open',
            'key_in_safe_lock','personal_item_on_desk','service_door_open','computer_screen_on','drink_over_cash']
    return {'t':list(map(float,range(n))), 'fps':1.0,'duration':float(n), 'p':{k:[0.0]*n for k in keys}}


class RegressionTests(unittest.TestCase):
    def test_persistence_is_rolling_minimum_and_rejects_spike_and_short_window(self):
        # state-min-v3: a state must hold on every frame of the window (mean-support v2 scored 62% vs 75%)
        self.assertAlmostEqual(se.sustain(np.array([.62,.41,.55]),3)[-1],.41)
        self.assertEqual(se.sustain(np.array([0.,.9,0.]),3)[-1],0.)
        self.assertEqual(se.sustain(np.array([.9,.1,.9]),3)[-1],.1)

    def test_cash_signal_is_elevation_over_the_video_own_level(self):
        f=facts();f['p']['owl_cash']=[.4]*16
        st=se.State(f)
        self.assertTrue(np.all(st.cash==0))      # a constant level is the background of this video, not an event

    def test_cash_unattended_time_is_not_before_departure(self):
        f=facts();f['p']['cash_on_desk']=[.8]*16
        f['p']['cashier_at_desk']=[.95]*6+[.01]*10
        f['p']['person_visible']=[.99]*6+[.01]*10
        events=se.evaluate(f,{'cash_unattended':.2},types=['cash_unattended'])['events']
        self.assertTrue(events)
        self.assertGreaterEqual(events[0]['timecode_sec'],6)

    def test_counting_alone_and_first_arrival_are_normal_contexts(self):
        f=facts();f['p']['cashier_at_desk']=[.95]*16;f['p']['person_visible']=[.99]*16
        f['p']['cash_in_hands']=[.9]*16;f['p']['cash_on_desk']=[.7]*16
        _, sc=se.type_scores(f)
        self.assertEqual(float(sc['counting_without_customer'][0].max()),0)
        f['p']['client_at_desk']=[0]*6+[.9]*10
        _, sc=se.type_scores(f)
        self.assertEqual(float(sc['previous_cash_on_table'][0].max()),0)

    def test_invalid_observations_fail_instead_of_returning_clean(self):
        for change in ('nan','length','time'):
            f=facts()
            if change=='nan': f['p']['cash_on_desk'][4]=float('nan')
            elif change=='length': f['p']['cash_on_desk'].pop()
            else: f['t'][5]=f['t'][4]
            with self.assertRaises(ValueError): se.State(f)

    def test_threshold_boundary_and_excluded_video_do_not_leak(self):
        self.assertEqual(choose_threshold([1,1,0,0],[.4,.8,.2,.5]),.4)
        fs={v:{'v_judge':{'type':'other','generic':True,'score':x,'t':1.0}}
            for v,x in {'p1':.4,'p2':.8,'test':.01,'n1':.2,'n2':.5}.items()}
        tr={v:{'label':'no_violation' if v.startswith('n') else 'violation','types':set()} for v in fs}
        fs['test']['heldout_only']={'type':'other','score':1,'t':0}
        cal=judge.calibrate(fs,tr,{'test'},rule='recall')
        self.assertEqual(cal['v_judge']['threshold'],.4)
        self.assertNotIn('test',cal[META]['training_videos'])
        self.assertNotIn('heldout_only',cal)
        self.assertEqual(cal[META]['calibration_misses'],[])
        self.assertEqual(cal[META]['calibration_false_alarms'],['n2'])
        with self.assertRaises(ValueError): choose_threshold([0,0],[.2,.3])

    def test_full_timeline_windows_include_the_end_of_long_video(self):
        st=se.State(facts(145));ws=flow_windows(st,145)
        self.assertEqual(ws[0][0],0)
        self.assertEqual(ws[-1][-1],144)
        self.assertGreater(len(ws),10)
        self.assertTrue(all(len(w)<=8 and w==sorted(set(w)) for w in ws))
        self.assertTrue(all(st.t[b[0]]-st.t[a[-1]]<=0 for a,b in zip(ws,ws[1:])))

    def test_same_name_changed_content_and_mock_switch_invalidate_cache(self):
        from pipeline.flow import _cache_identity
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);v=d/'same.mov';v.write_bytes(b'first');cache=d/'cache'
            _cache_identity(str(v),cache)
            old=cache/'facts'/'same.json';old.parent.mkdir();old.write_text('{}')
            _cache_identity(str(v),cache);self.assertTrue(old.exists())
            v.write_bytes(b'second');_cache_identity(str(v),cache);self.assertFalse(old.exists())
            old.write_text('{}');_cache_identity(str(v),cache,mock=True);self.assertFalse(old.exists())

    def test_saved_policy_is_frozen_and_incompatible_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'policy.json'
            cal={'v_judge':{'enabled':True,'source':'vlm','threshold':.5,'generic':True}}
            judge.save(cal,{},p)
            before=p.read_bytes()
            self.assertEqual(judge.load(p),cal)
            with patch.dict('os.environ',{'FACT_FPS':'2'}):
                with self.assertRaisesRegex(RuntimeError,'settings differ'):judge.load(p)
            self.assertEqual(before,p.read_bytes())
            p.write_text(json.dumps({'requires_calibration':True}))
            with self.assertRaisesRegex(RuntimeError,'requires calibration'):judge.load(p)

    def test_inference_does_not_reuse_training_facts_and_api_contract_is_valid(self):
        import cv2
        from pipeline import flow
        from pipeline.mock import MockVLM
        import pipeline.analyze as api
        class Spy(MockVLM):
            calls=0
            def frame_question_probs(self,*a,**kw):
                self.calls+=1
                return super().frame_question_probs(*a,**kw)
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);v=d/'X-1.mp4';w=cv2.VideoWriter(str(v),cv2.VideoWriter_fourcc(*'mp4v'),1,(160,240))
            self.assertTrue(w.isOpened())
            for i in range(16):w.write(np.full((240,160,3),40+i*5,dtype=np.uint8))
            w.release();legacy=d/'training';legacy.mkdir();(legacy/'X-1.json').write_text(json.dumps(facts()))
            m=flow.Models(mock=True);spy=Spy();m._vlm=spy
            r=flow.process(str(v),m,cache=None,facts_dirs=(legacy,),run_chains=False,run_change=False)
            self.assertGreater(spy.calls,0)
            self.assertEqual(r['rec']['classifier_version'],'classify-flow-v2')
            self.assertTrue(r['rec']['mock'])
            cal={'v_judge':{'enabled':True,'source':'vlm','threshold':0.0,'clean_max':1.0,'generic':True},
                 META:{'enabled':False,'source':'policy'}}
            before=copy.deepcopy(cal)
            ev=judge.decide(r['rec']['features'],r['facts'],cal)
            self.assertTrue(ev);self.assertEqual(before,cal)
            with patch.object(judge,'load',return_value=cal),patch.object(api,'_MODELS',m),patch.object(api,'DESCRIBE',False):
                output=api.analyze(str(v))
            self.assertTrue(output['violations'])
            for event in output['violations']:
                self.assertTrue({'type','timecode_sec','description','evidence_frames'}<=event.keys())
                self.assertTrue(0<=event['timecode_sec']<16)
                self.assertTrue(1<=len(event['evidence_frames'])<=3)
                for image in event['evidence_frames']:
                    raw=base64.b64decode(image['image_base64']);self.assertTrue(raw.startswith(b'\xff\xd8'))

if __name__=='__main__':unittest.main()
