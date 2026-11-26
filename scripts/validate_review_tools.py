"""CPU implementation evidence; synthetic shift truth never enters the tools."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lagent.schema import ActionStatusCode
from tools.demo_review_tools_agent import make_executor
from tools.review_tool_utils import sha256,write_json


def invoke(executor,name,a,b,out):
    action=executor(name,json.dumps(dict(image_before=str(a),image_after=str(b),output_dir=str(out))))
    if action.state!=ActionStatusCode.SUCCESS:
        raise RuntimeError(name+': '+str(action.errmsg))
    return json.loads(action.result[0]['content'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/review-tools-validation-20261123')
    parser.add_argument('--real-data',type=Path,default=ROOT/'work/cdd-building-eval-20261118/test')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():raise ValueError('Choose a new validation directory; do not overwrite evidence')
    output.mkdir(parents=True)
    inputs,artifacts=output/'inputs',output/'tool_runs'
    inputs.mkdir()
    # Known transform is stored solely in this offline evaluation specification.
    rng=np.random.default_rng(54)
    rgb=cv2.GaussianBlur(rng.integers(15,240,(256,256,3),dtype=np.uint8),(5,5),.7)
    changed=rgb.copy();changed[96:128,112:152]=[245,35,45]
    cases=[('identity',rgb,rgb,(0,0)),('integer_positive',rgb,rgb,(4,-3)),
           ('integer_negative',rgb,rgb,(-5,4)),('subpixel',rgb,rgb,(2.4,-3.6)),
           ('true_change_no_shift',rgb,changed,(0,0)),('true_change_and_shift',rgb,changed,(4,-3)),
           ('too_large',rgb,rgb,(40,0)),
           ('constant',np.full_like(rgb,100),np.full_like(rgb,100),(0,0)),
           ('unrelated',rgb,rng.integers(0,256,rgb.shape,dtype=np.uint8),(0,0)),
           ('blurred',rgb,cv2.GaussianBlur(rgb,(0,0),1.5),(0,0))]
    records=[]
    visual=[]
    for name,a,b,shift in cases:
        moving=cv2.warpAffine(b,np.float32([[1,0,shift[0]],[0,1,shift[1]]]),(256,256),
                              flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101) if shift!=(0,0) else b.copy()
        pa,pb=inputs/(name+'_A.png'),inputs/(name+'_B.png')
        Image.fromarray(a).save(pa);Image.fromarray(moving).save(pb)
        original=(sha256(pa),sha256(pb))
        executor=make_executor()
        quality=invoke(executor,'image_quality_check',pa,pb,artifacts)
        check=invoke(executor,'alignment_check',pa,pb,artifacts)
        corrected=invoke(executor,'registration_correct',pa,pb,artifacts)
        status=corrected['diagnostics']['status']
        row=dict(case=name,known_displacement_reference_to_moving_px=list(shift),
                 quality_status=quality['diagnostics']['assessment'],
                 alignment_status=check['diagnostics']['status'],registration_status=status,
                 reason=corrected['diagnostics']['reason'],record_path=corrected['artifacts'][-1]['path'])
        if status in ('registered','noop'):
            matrix=corrected['diagnostics']['transform']['sampling_matrix_reference_to_moving']
            estimated=np.asarray(matrix)[:,2]
            row['known_transform_error_px']=float(np.linalg.norm(estimated-np.asarray(shift)))
            files={item['kind']:item['path'] for item in corrected['artifacts']}
            registered=np.asarray(Image.open(files['registered_image']))
            valid=np.asarray(Image.open(files['valid_mask'])).astype(bool)
            row['valid_fraction']=float(valid.mean())
            row['reference_target_mae_on_valid']=float(np.mean(np.abs(registered[valid].astype(float)-b[valid].astype(float))))
            if name in ('identity','true_change_no_shift'):
                assert status=='noop' and np.array_equal(registered,b),name
            if name in ('integer_positive','integer_negative','subpixel','true_change_and_shift'):
                assert row['known_transform_error_px']<.3,name
            if name=='true_change_and_shift':
                roi=np.zeros((256,256),bool);roi[96:128,112:152]=True
                row['changed_patch_mae_from_true_changed_B']=float(np.abs(registered.astype(float)-b.astype(float))[roi&valid].mean())
                assert row['changed_patch_mae_from_true_changed_B']<2,name
            if name in ('integer_positive','true_change_and_shift','subpixel'):
                visual.append((name,a,moving,registered,valid))
        if name in ('constant','unrelated','too_large'):
            assert status in ('unknown','refused'),(name,status)
        assert original==(sha256(pa),sha256(pb)),name
        records.append(row)
    real=[]
    if args.real_data.is_dir():
        names=sorted((args.real_data/'A').glob('*.png'))[:20]
        for index,pa in enumerate(names):
            pb=inputs/('real_shift_'+pa.name)
            image=np.asarray(Image.open(pa)).copy()
            dx,dy=(4,-3) if index%2==0 else (-4,3)
            moving=cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,dy]]),(256,256),borderMode=cv2.BORDER_REFLECT_101)
            Image.fromarray(moving).save(pb)
            result=invoke(make_executor(),'registration_correct',pa,pb,artifacts)
            status=result['diagnostics']['status']
            row=dict(name=pa.name,status=status,reason=result['diagnostics']['reason'],known_shift_px=[dx,dy],
                     record_path=result['artifacts'][-1]['path'])
            if status in ('registered','noop'):
                matrix=np.asarray(result['diagnostics']['transform']['sampling_matrix_reference_to_moving'])
                row['known_transform_error_px']=float(np.linalg.norm(matrix[:,2]-[dx,dy]))
                assert row['known_transform_error_px']<.5,row
            real.append(row)
    # Deliberately explicit status: real imagery with synthetic known geometry,
    # not independent checkpoints for naturally occurring temporal misregistration.
    summary=dict(validation_type='implementation/contract + known synthetic shift on real imagery',
                 offline_synthetic_cases=records,real_source_known_shift_cases=real,
                 limits=['No ground truth or perturbation parameter passed to any tool',
                         'Heuristic thresholds are uncalibrated; no real-world registration accuracy or adaptive policy claim',
                         'No downstream detector was invoked on invalid borders',
                         'ActionExecutor calls verified; real DeepSeek invocation requires the user API key'],
                 schema_version='0.1')
    errors=[r['known_transform_error_px'] for r in real if 'known_transform_error_px' in r]
    summary['real_source_summary']=dict(total=len(real),accepted=len(errors),
        statuses={s:sum(r['status']==s for r in real) for s in sorted({r['status'] for r in real})},
        median_known_transform_error_px=float(np.median(errors)) if errors else None,
        max_known_transform_error_px=max(errors) if errors else None)
    write_json(output/'validation.json',summary)
    if visual:
        try:
            font=ImageFont.truetype('Arial.ttf',15)
        except OSError:
            font=ImageFont.load_default()
        board=Image.new('RGB',(4*256,len(visual)*286+30),'white');draw=ImageDraw.Draw(board)
        for col,label in enumerate(('Reference A','Displaced B','B corrected to A','Valid mask (white=1)')):
            draw.text((col*256+5,5),label,fill='black',font=font)
        for row,(name,a,b,corrected,valid) in enumerate(visual):
            y=30+row*286
            for col,array in enumerate((a,b,corrected,np.repeat((valid.astype(np.uint8)*255)[:,:,None],3,axis=2))):
                board.paste(Image.fromarray(array),(col*256,y))
            draw.text((5,y+260),name+' | truth used only in offline evaluator',fill='black',font=font)
        board.save(output/'registration_evidence.png')
    print(json.dumps(dict(output=str(output),synthetic_cases=len(records),**summary['real_source_summary']),ensure_ascii=False))


if __name__=='__main__':main()
