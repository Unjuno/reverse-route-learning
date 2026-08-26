#!/usr/bin/env python3
"""Fixed-protocol TinyStories-8M capability-boundary benchmark used in v0.1."""
import argparse, copy, json, math, time
from pathlib import Path
import torch
import torch.nn.functional as F
from reverse_route_learning.runtime import TinyStoriesNeo, kv_prefix_cache, kv_step

CONDITIONS = [
    (49, 8, 'shallow_reconvergent'),
    (43, 8, 'shallow_reconvergent'),
    (17, 8, 'shallow_divergent'),
    (19, 8, 'shallow_divergent'),
    (19, 20, 'deep_eroded'),
    (31, 20, 'deep_eroded'),
    (36, 20, 'deep_eroded'),
    (54, 8, 'shallow_reconvergent'),
    (0, 8, 'shallow_reconvergent'),
]

def js_logits(a,b):
    la=F.log_softmax(a,-1); lb=F.log_softmax(b,-1)
    pa=la.exp(); pb=lb.exp(); m=.5*(pa+pb); lm=m.clamp_min(1e-30).log()
    return .5*((pa*(la-lm)).sum(-1)+(pb*(lb-lm)).sum(-1))

def train_post(model_path, c, steps, lr=4e-6, suffix=24):
    base=TinyStoriesNeo(model_path).eval(); pref=c['prefix_ids']; A=c['A_id']
    seq=base.greedy(pref+[A],suffix)
    post=TinyStoriesNeo(model_path).train(); opt=torch.optim.AdamW(post.parameters(),lr=lr,weight_decay=0)
    ids=torch.tensor(seq,dtype=torch.long)[None,:]
    for _ in range(steps):
        opt.zero_grad(); z=post(ids[:,:-1]); tg=ids[0,1:]
        loss=F.cross_entropy(z[0,len(pref)-1:],tg[len(pref)-1:]); loss.backward(); opt.step()
    return base,post.eval()

@torch.no_grad()
def scan(post,c,K=32,H=12):
    pref=c['prefix_ids']; z0=post.last_logits(torch.tensor(pref,dtype=torch.long))[0]; p0=F.softmax(z0,-1)
    top=torch.topk(p0,K+1).indices.tolist(); default=int(top[0]); cand=[int(x) for x in top[1:]]
    pc=kv_prefix_cache(post,pref)
    rz,rc,rh=kv_step(post,pc,[default],return_hidden=True); refs=[]
    for d in range(H):
        refs.append(rz[0].clone())
        if d<H-1: rz,rc,rh=kv_step(post,rc,rz.argmax(-1),return_hidden=True)
    z,cache,h=kv_step(post,pc,torch.tensor(cand),return_hidden=True)
    stats={t:{'js':[],'nll':[],'ent':[]} for t in cand}
    for d in range(H):
        lp=F.log_softmax(z,-1); p=lp.exp(); nt=z.argmax(-1); nll=-lp.gather(1,nt[:,None]).squeeze(1); ent=-(p*lp).sum(-1)
        js=js_logits(z,refs[d][None,:].expand(len(cand),-1))
        for j,t in enumerate(cand):
            stats[t]['js'].append(float(js[j])); stats[t]['nll'].append(float(nll[j])); stats[t]['ent'].append(float(ent[j]))
        if d<H-1: z,cache,h=kv_step(post,cache,nt,return_hidden=True)
    rows=[]
    for t in cand:
        s=stats[t]; rows.append({'id':t,'entry_rank':int((z0>z0[t]).sum())+1,'late_js_min':min(s['js'][1:]),'mean_self_nll':sum(s['nll'])/H,'mean_entropy':sum(s['ent'])/H})
    for key,rev in [('late_js_min',False),('mean_self_nll',False),('mean_entropy',True)]:
        for rank,r in enumerate(sorted(rows,key=lambda x:x[key],reverse=rev),1): r['rank_'+key]=rank
    for r in rows: r['intrinsic_score']=r['entry_rank']+min(r['rank_mean_self_nll'],r['rank_mean_entropy'])
    for rank,r in enumerate(sorted(rows,key=lambda x:x['intrinsic_score']),1): r['rank_intrinsic']=rank
    selected=[]
    for r in sorted(rows,key=lambda x:x['late_js_min'])[:3]+sorted(rows,key=lambda x:x['intrinsic_score'])[:3]:
        if r['id'] not in selected: selected.append(r['id'])
    return rows,selected,p0

def repair(post,c,selected,p0,kl_budget=.05,lr=1e-6,max_steps=30):
    pref=torch.tensor(c['prefix_ids'],dtype=torch.long)[None,:]; ids=torch.tensor(selected,dtype=torch.long)
    m=copy.deepcopy(post).train(); opt=torch.optim.SGD(m.parameters(),lr=lr); kl=0.; steps=0
    for step in range(max_steps):
        opt.zero_grad(); lp=F.log_softmax(m(pref)[0,-1],-1); (-torch.logsumexp(lp[ids],0)).backward(); opt.step(); steps=step+1
        with torch.no_grad():
            pf=F.softmax(m(pref)[0,-1],-1); kl=float((p0*(p0.clamp_min(1e-30).log()-pf.clamp_min(1e-30).log())).sum())
        if kl>=kl_budget: break
    return m.eval(),steps,kl

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--branches',type=Path,default=Path('data/tinystories_capability_boundary_branches.json')); ap.add_argument('--output',type=Path,default=Path('results/tinystories/repro_capability_boundary.json')); ap.add_argument('--threads',type=int,default=4); args=ap.parse_args()
    torch.set_num_threads(args.threads); branches=json.loads(args.branches.read_text()); byid={x['original_case_id']:x for x in branches}; out=[]
    for cid,sft,label in CONDITIONS:
        t=time.time(); c=byid[cid]; base,post=train_post(args.model,c,sft); rows,selected,p0=scan(post,c); B=c['B_id']; brow=next((r for r in rows if r['id']==B),None)
        pref=torch.tensor(c['prefix_ids'],dtype=torch.long)
        with torch.no_grad():
            bp=float(F.softmax(base.last_logits(pref)[0],-1)[B]); pp=float(F.softmax(post.last_logits(pref)[0],-1)[B])
        bseq=base.greedy(c['prefix_ids']+[B],16); ids=torch.tensor(bseq,dtype=torch.long)[None,:]
        with torch.no_grad(): lb=F.log_softmax(base(ids[:,:-1])[0],-1); lp=F.log_softmax(post(ids[:,:-1])[0],-1); tg=ids[0,1:]
        ds=[float(lp[j,tg[j]]-lb[j,tg[j]]) for j in range(len(c['prefix_ids']),len(tg))]; support=math.exp(sum(ds)/max(1,len(ds)))
        repaired,nstep,kl=repair(post,c,selected,p0)
        with torch.no_grad(): fp=float(F.softmax(repaired.last_logits(pref)[0],-1)[B])
        rec=(fp-pp)/(bp-pp) if abs(bp-pp)>1e-12 else float('nan')
        r={'case':cid,'label':label,'sft_steps':sft,'A':c['A_text'],'B':c['B_text'],'closure':bp/pp,'suffix_support':support,'B_in_top32':brow is not None,'B_rank_reconvergence':None if brow is None else brow['rank_late_js_min'],'B_rank_intrinsic':None if brow is None else brow['rank_intrinsic'],'B_selected':B in selected,'oracle_recovery_fraction':rec,'entry_KL':kl,'repair_steps':nstep,'seconds':time.time()-t}; out.append(r); print(r,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2))
if __name__=='__main__': main()
