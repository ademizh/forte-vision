"""Behavioral regression tests. Scripted vision answers are not quality measurements."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import focus,judge,flow,state_engine as se
from pipeline.frames import Frame
from pipeline.recall_policy import META
from pipeline.temporal import accumulate
from test_recall_patch import facts


class DescriptiveVLM:
    """Answer by semantic state, regardless of which letter currently names it."""
    def __init__(self,positive_times=()):self.positive_times=set(positive_times);self.calls=0
    def frame_question_probs(self,system,frames,questions,n_options):
        self.calls+=1;result=[]
        for f in frames:
            answers=[]
            for question in questions:
                condition=next(k for k,v in focus.CONDITIONS.items() if all(x in question for x in v))
                positive=condition in ('key_in_lock','away_safe','safe_open') and round(f.t) in self.positive_times
                semantic=[.9,.05,.05] if positive else [.01,.98,.01]
                option_lines=[x[3:] for x in question.splitlines() if len(x)>2 and x[1:3]=='. ']
                answers.append([semantic[focus.CONDITIONS[condition].index(text)] for text in option_lines])
            result.append(answers)
        return result


def fixture(n=12,weak_box=True):
    fa=facts(n);fa['p']['cashier_at_desk']=[.8]*n;fa['p']['person_visible']=[.95]*n
    st=se.State(fa)
    frames=[Frame(float(i),Image.new('RGB',(160,240),(40+i*3,40,40))) for i in range(n)]
    per=[{'key':[[.04,10,20,30,50]]} if weak_box else {} for _ in range(n)]
    det={'t':fa['t'],'size':[160,240],'dets':{'owl':per},'tiles':{'owl':[None]*n}}
    class Scene:rois={};tracks={};n=0
    vc={'features':{'v_judge':{'type':'other','score':.2,'t':3.}}}
    return fa,st,frames,det,Scene(),vc


class SoftFilterTests(unittest.TestCase):
    def test_three_of_five_uses_first_positive_and_retains_brief(self):
        out=accumulate(range(7),[0,.8,0,.7,.9,0,0])
        self.assertEqual(out['events'][0]['start'],1)
        self.assertAlmostEqual(out['events'][0]['score'],.8)
        one=accumulate(range(7),[0,0,0,.9,0,0,0])
        self.assertEqual(one['events'],[])
        self.assertEqual(one['brief'][0]['start'],3)

    def test_separate_incidents_and_missing_timestamps_do_not_merge(self):
        s=np.zeros(16);s[1:4]=.9;s[10:13]=.8
        out=accumulate(range(16),s)
        self.assertEqual([e['start'] for e in out['events']],[1.,10.])
        self.assertFalse(accumulate([0,1,10],[.8,.8,.8],max_gap=1.75)['events'])
        with self.assertRaises(ValueError):accumulate([0,0,1],[.5,.5,.5])

    def test_raw_weak_owl_box_bypasses_high_track_gate(self):
        _,st,_,det,scene,vc=fixture()
        self.assertEqual(focus.low_proposals(det,('key',),1)[0]['score'],.04)
        candidates,_=focus.collect(st,scene,det,vc)
        self.assertIn('safe',[c['group'] for c in candidates])
        self.assertTrue(any(c['box'] is not None for c in candidates))

    def test_no_owl_detection_still_receives_full_frame_survey(self):
        _,st,fr,det,scene,vc=fixture(weak_box=False)
        candidates,_=focus.collect(st,scene,det,vc)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0]['proposal_source'],'independent full-frame survey')
        model=DescriptiveVLM([2,3,4]);r=focus.run(st,scene,det,fr,model,vc)
        self.assertGreater(model.calls,0)
        self.assertIn('f_safe_key_unattended',r['features'])

    def test_descriptive_token_scores_are_letter_order_invariant(self):
        _,_,fr,_,_,_=fixture()
        p=focus.state_probabilities(DescriptiveVLM([2]),[fr[2],fr[3]],['key_in_lock'])['key_in_lock']
        np.testing.assert_allclose(p,[[.9,.05,.05],[.01,.98,.01]])

    def test_negative_description_containing_key_does_not_trigger_keyword_alarm(self):
        _,st,fr,det,scene,vc=fixture()
        r=focus.run(st,scene,det,fr,DescriptiveVLM(),vc)
        self.assertEqual(r['events'],[])
        self.assertEqual(r['features']['f_any']['score'],0)

    def test_temporal_type_votes_and_onset_are_preserved(self):
        _,st,fr,det,scene,vc=fixture()
        r=focus.run(st,scene,det,fr,DescriptiveVLM([2,3,4]),vc)
        for typ in ('safe_key_unattended','safe_left_open'):
            e=r['features']['f_'+typ]
            self.assertTrue(e['type_supported']);self.assertEqual(e['t'],2)
            self.assertGreaterEqual(e['support_count'],3)

    def test_overlapping_views_are_not_extra_temporal_votes(self):
        _,st,fr,det,scene,vc=fixture()
        r=focus.run(st,scene,det,fr,DescriptiveVLM([3]),vc)
        self.assertFalse(any(e['type_supported'] for e in r['events']+r['brief']))
        self.assertTrue(r['brief'])
        self.assertTrue(all(e['support_count']==1 for e in r['brief']))

    def test_independent_branch_can_add_event_despite_low_global_score(self):
        fa,st,fr,det,scene,vc=fixture();r=focus.run(st,scene,det,fr,DescriptiveVLM([2,3,4]),vc)
        fs={**r['features'],'v_judge':{'type':'other','score':.01,'t':3.}}
        cal={'v_judge':{'enabled':True,'source':'vlm','threshold':.8,'generic':True,'clean_max':.2},
             'f_safe_key_unattended':{'enabled':True,'source':'focus','threshold':.7,'generic':False,'clean_max':.2},
             META:{'enabled':False,'source':'policy'}}
        before=copy.deepcopy(cal);events=judge.decide(fs,fa,cal)
        self.assertTrue(any(e['id']=='safe_key_unattended' for e in events));self.assertEqual(cal,before)
        fs.pop('f_any')
        with self.assertRaisesRegex(ValueError,'branch did not run'):judge.decide(fs,fa,cal)

    def test_clear_crop_is_not_averaged_away_by_blurry_full_frame(self):
        class ViewModel(DescriptiveVLM):
            def frame_question_probs(self,system,frames,questions,n_options):
                rows=[]
                for frame in frames:
                    model=DescriptiveVLM([2,3,4] if frame.image.width>160 else [])
                    rows.extend(model.frame_question_probs(system,[frame],questions,n_options))
                return rows
        _,st,fr,det,scene,vc=fixture()
        det['dets']['owl'][3]['key'][0][0]=.10
        r=focus.run(st,scene,det,fr,ViewModel(),vc)
        event=r['features']['f_safe_key_unattended']
        self.assertTrue(event['type_supported'])
        self.assertAlmostEqual(event['score'],.9)

    def test_negative_focus_does_not_veto_global_alarm(self):
        fa,_,_,_,_,_=fixture()
        fs={'v_judge':{'type':'other','score':.9,'t':3.,'generic':True},
            'f_any':{'type':'other','score':0.,'t':0.,'episodes':[]}}
        cal={'v_judge':{'enabled':True,'source':'vlm','threshold':.7,'generic':True,'clean_max':.5},
             'f_any':{'enabled':True,'source':'focus','threshold':.5,'generic':True,'clean_max':.2},
             META:{'enabled':False,'source':'policy'}}
        self.assertTrue(judge.decide(fs,fa,cal))

    def test_joint_cutoffs_reduce_false_alarms_without_calibration_misses(self):
        values={'p1':(.4,.9),'p2':(.8,.1),'n1':(.6,.1),'n2':(.3,.5)}
        fs={v:{'v_judge':{'type':'other','score':a,'t':0.,'generic':True},
               'f_any':{'type':'other','score':b,'t':0.,'generic':True}} for v,(a,b) in values.items()}
        truth={v:{'label':'violation' if v.startswith('p') else 'no_violation','types':set()} for v in values}
        cal=judge.calibrate(fs,truth,rule='recall')
        self.assertEqual(cal[META]['calibration_misses'],[])
        self.assertEqual(cal[META]['calibration_false_alarms'],[])
        self.assertEqual(cal['v_judge']['threshold'],.8)
        self.assertEqual(cal['f_any']['threshold'],.9)
        self.assertEqual(len(cal[META]['selected']),2)

    def test_distinct_episodes_of_same_type_survive_output_deduplication(self):
        fa,_,_,_,_,_=fixture(20)
        def e(t):return {'type':'safe_key_unattended','score':.9,'t':t,'start':t,'end':t+2,
                         'persistent':True,'type_supported':True,'why':'visible state'}
        fs={'f_safe_key_unattended':{**e(1),'episodes':[e(1),e(10)]},
            'f_any':{'type':'other','score':0,'t':0,'episodes':[]}}
        cal={'f_safe_key_unattended':{'enabled':True,'source':'focus','threshold':.7,'generic':False,'clean_max':.2},
             META:{'enabled':False,'source':'policy'}}
        events=judge.decide(fs,fa,cal)
        self.assertEqual(sorted(e['timecode_sec'] for e in events),[1,10])

    def test_focus_settings_change_keeps_existing_base_cache(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);video=d/'x.mov';video.write_bytes(b'video');cache=d/'cache'
            flow._cache_identity(str(video),cache)
            for sub in ('det','facts','vlm'):
                p=cache/sub/'x.json';p.parent.mkdir(parents=True);p.write_text('{}')
            sig=focus.signature();sig['max_windows']+=1
            with patch.object(focus,'signature',return_value=sig):flow._cache_identity(str(video),cache)
            self.assertTrue(all((cache/sub/'x.json').exists() for sub in ('det','facts','vlm')))
            self.assertNotIn('focus',json.loads((cache/'identity'/'x.json').read_text())['inference_signature'])

    def test_crop_contains_context_and_clamps_invalid_box(self):
        im=Image.new('RGB',(160,240),'gray')
        view=focus.detail_image(im,[10,20,30,50])
        self.assertGreater(view.width,im.width)
        self.assertEqual(focus.detail_image(im,[float('nan'),0,1,2]).size,im.size)

    def test_focus_upgrade_reuses_owl_facts_and_whole_flow_without_new_base_calls(self):
        import cv2
        from pipeline.mock import MockVLM
        class Spy(MockVLM):
            frame_calls=0;choice_calls=0
            def frame_question_probs(self,*a,**k):
                self.frame_calls+=1;return super().frame_question_probs(*a,**k)
            def choice_probs(self,*a,**k):
                self.choice_calls+=1;return super().choice_probs(*a,**k)
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);v=d/'x.mp4';cache=d/'cache'
            writer=cv2.VideoWriter(str(v),cv2.VideoWriter_fourcc(*'mp4v'),1,(160,240))
            self.assertTrue(writer.isOpened())
            for i in range(12):writer.write(np.full((240,160,3),30+5*i,dtype=np.uint8))
            writer.release()
            first=flow.Models(mock=True)
            flow.process(str(v),first,cache,run_chains=False,run_change=False,run_focus=False)
            spy=Spy();second=flow.Models(mock=True);second._vlm=spy
            result=flow.process(str(v),second,cache,run_chains=False,run_change=False,run_focus=True)
            self.assertGreater(spy.frame_calls,0);self.assertEqual(spy.choice_calls,0)
            self.assertNotIn('owl',result['rec']['timing'])
            self.assertNotIn('qwen_facts',result['rec']['timing'])
            self.assertNotIn('classify',result['rec']['timing'])
            self.assertIn('focus',result['rec']['timing'])
            self.assertEqual(result['focus']['version'],focus.VERSION)

    def test_focused_api_output_keeps_valid_frames_and_first_supported_time(self):
        import base64
        import cv2
        from pipeline.mock import MockVLM
        import pipeline.analyze as api
        class StatesWithBase(MockVLM):
            def frame_question_probs(self,system,frames,questions,n_options=2):
                if n_options==3:
                    return DescriptiveVLM(range(20)).frame_question_probs(system,frames,questions,n_options)
                return super().frame_question_probs(system,frames,questions,n_options)
        with tempfile.TemporaryDirectory() as td:
            v=Path(td)/'jury_new_name.mp4';writer=cv2.VideoWriter(str(v),cv2.VideoWriter_fourcc(*'mp4v'),1,(160,240))
            self.assertTrue(writer.isOpened())
            for i in range(12):writer.write(np.full((240,160,3),70+i,dtype=np.uint8))
            writer.release();m=flow.Models(mock=True);m._vlm=StatesWithBase()
            cal={'f_any':{'source':'focus','enabled':True,'threshold':.5,'clean_max':1.,'generic':True},
                 META:{'source':'policy','enabled':False}}
            with patch.object(judge,'load',return_value=cal),patch.object(api,'_MODELS',m),patch.object(api,'DESCRIBE',False):
                output=api.analyze(str(v))
            self.assertTrue(output['violations'])
            for e in output['violations']:
                self.assertTrue(0<=e['timecode_sec']<12)
                self.assertTrue(1<=len(e['evidence_frames'])<=3)
                self.assertTrue(all(base64.b64decode(f['image_base64']).startswith(b'\xff\xd8') for f in e['evidence_frames']))

    def test_force_dependencies_include_focused_analysis(self):
        self.assertTrue({'chain','vlm','focus'}<=flow.expand_force({'facts'}))
        self.assertEqual(flow.expand_force({'focus'}),{'focus'})

if __name__=='__main__':unittest.main()
