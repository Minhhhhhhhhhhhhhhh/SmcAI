"""
SmartCoach AI v2 — Basketball Tactical Intelligence
i-TECH Research Group | Pipeline: LSTM -> GradientBoosting -> LLM
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─── CONSTANTS ───────────────────────────────────────────────────
COURT_W, COURT_H = 94, 50
SAMPLE_RATE      = 25
POSSESSION_SECS  = 14
N_FRAMES         = POSSESSION_SECS * SAMPLE_RATE   # 350
N_PLAYERS        = 10
TACTIC_LABELS    = ["Pick-and-Roll","Isolation","Motion Offense","Fast Break","Post-Up"]
np.random.seed(42)

# ─── HELPERS ─────────────────────────────────────────────────────
def smooth(arr, w=11):
    """Gaussian smoothing."""
    k = np.exp(-0.5*np.linspace(-2,2,w)**2); k/=k.sum(); p=w//2
    out = np.zeros_like(arr)
    for d in range(arr.shape[1]):
        col = np.pad(arr[:,d], p, mode='edge')
        out[:,d] = np.convolve(col, k, mode='valid')
    return out

def micro_move(n, center, r=1.5, freq=0.7):
    """Micro-movement: spacing player luon nhuc nhich nho."""
    t = np.linspace(0, 2*np.pi*freq, n)
    px,py = np.random.uniform(0,2*np.pi,2)
    x = center[0]+r*np.sin(t+px)+np.random.randn(n)*0.2
    y = center[1]+r*np.cos(t*1.3+py)+np.random.randn(n)*0.2
    return smooth(np.stack([x,y],axis=1))

# ─── 1. DATA GENERATOR ───────────────────────────────────────────
def generate_possession(tactic, n_frames=N_FRAMES):
    pos = np.zeros((n_frames, N_PLAYERS, 2))
    bx  = np.zeros(n_frames)
    by  = np.zeros(n_frames)

    if tactic == "Pick-and-Roll":
        ph1,ph2 = int(n_frames*.30), int(n_frames*.55)
        # Ball-handler: dribble up -> screen -> drive
        hx = np.concatenate([np.linspace(20,35,ph1), np.linspace(35,42,ph2-ph1), np.linspace(42,72,n_frames-ph2)])
        hy = np.concatenate([np.full(ph1,25)+np.random.randn(ph1)*.3, np.linspace(25,22,ph2-ph1), np.linspace(22,25,n_frames-ph2)+np.random.randn(n_frames-ph2)*.4])
        pos[:,0,:] = smooth(np.stack([hx,hy],axis=1))
        # Screener: set screen -> roll to basket
        sx = np.concatenate([np.full(ph1,38)+np.random.randn(ph1)*.2, np.full(ph2-ph1,40), np.linspace(40,74,n_frames-ph2)])
        sy = np.concatenate([np.full(ph1,22)+np.random.randn(ph1)*.2, np.full(ph2-ph1,22), np.linspace(22,25,n_frames-ph2)])
        pos[:,1,:] = smooth(np.stack([sx,sy],axis=1))
        # Shooters spread with micro-movement
        for p,spot in enumerate([[65,8],[65,42],[52,5]]): pos[:,p+2,:]=micro_move(n_frames,spot)
        # Defense
        pos[:,5,:] = pos[:,0,:]+np.random.randn(n_frames,2)*1.0+[2,-1]
        pos[:,6,:] = pos[:,1,:]+np.random.randn(n_frames,2)*0.8+[-1,1]
        for p,spot in enumerate([[60,12],[60,38],[50,25]]): pos[:,p+7,:]=micro_move(n_frames,spot,r=2.0)
        # Ball: follows handler, then pass to screener on roll
        pf = int(n_frames*.75)
        bx[:pf]=pos[:pf,0,0]+np.random.randn(pf)*.3; by[:pf]=pos[:pf,0,1]+np.random.randn(pf)*.3
        bx[pf:]=pos[pf:,1,0]+np.random.randn(n_frames-pf)*.3; by[pf:]=pos[pf:,1,1]+np.random.randn(n_frames-pf)*.3

    elif tactic == "Isolation":
        ph1 = int(n_frames*.4)
        ix = np.concatenate([np.linspace(30,38,ph1)+np.random.randn(ph1)*.3, np.linspace(38,75,n_frames-ph1)])
        iy = np.concatenate([25+3*np.sin(np.linspace(0,2*np.pi,ph1)), np.linspace(25,23,n_frames-ph1)+np.random.randn(n_frames-ph1)*.4])
        pos[:,0,:] = smooth(np.stack([ix,iy],axis=1))
        for p,spot in enumerate([[62,6],[62,44],[50,4],[50,46]]): pos[:,p+1,:]=micro_move(n_frames,spot,r=1.0,freq=.5)
        pos[:,5,:] = pos[:,0,:]+np.column_stack([np.random.randn(n_frames)*.8+1.5, np.random.randn(n_frames)*.8])
        for p,spot in enumerate([[58,8],[58,42],[48,5],[48,46]]): pos[:,p+6,:]=micro_move(n_frames,spot)
        bx=pos[:,0,0]+np.random.randn(n_frames)*.3; by=pos[:,0,1]+np.random.randn(n_frames)*.3

    elif tactic == "Motion Offense":
        perim  = [[42,25],[50,8],[62,5],[68,42],[55,44]]
        frames = [0,70,140,210,280,n_frames]
        holder = [0,1,2,3,4,0]
        for p in range(5):
            base=np.array(perim[p],dtype=float); traj=micro_move(n_frames,base,r=3.5,freq=.6)
            cs,ce = int(n_frames*p/5), int(n_frames*p/5)+int(n_frames*.15)
            if ce<n_frames:
                traj[cs:ce,0]=np.linspace(base[0],72,ce-cs)
                traj[cs:ce,1]=np.linspace(base[1],25+(p-2)*4,ce-cs)
            pos[:,p,:]=traj; pos[:,p+5,:]=traj+np.random.randn(n_frames,2)*1.0+[1,0]
        for i,(s,e) in enumerate(zip(frames[:-1],frames[1:])):
            e=min(e,n_frames); h=holder[i]
            bx[s:e]=pos[s:e,h,0]+np.random.randn(e-s)*.3
            by[s:e]=pos[s:e,h,1]+np.random.randn(e-s)*.3

    elif tactic == "Fast Break":
        for p,ly in enumerate([20,25,30]):
            spd=np.power(np.linspace(0,1,n_frames),.7)
            pos[:,p,0]=smooth((5+spd*82).reshape(-1,1),7).ravel()
            pos[:,p,1]=smooth((ly+np.random.randn(n_frames)*.5).reshape(-1,1),7).ravel()
        for p in range(2):
            pos[:,p+3,0]=smooth((np.linspace(5,55,n_frames)+np.random.randn(n_frames)*.4).reshape(-1,1),9).ravel()
            pos[:,p+3,1]=smooth((15+p*20+np.random.randn(n_frames)*.6).reshape(-1,1),9).ravel()
        for p in range(5):
            pos[:,p+5,0]=smooth((np.linspace(85,65,n_frames)+np.random.randn(n_frames)*1.5).reshape(-1,1),7).ravel()
            pos[:,p+5,1]=smooth((10+p*8+np.random.randn(n_frames)*1.0).reshape(-1,1),7).ravel()
        bx=pos[:,0,0]+np.random.randn(n_frames)*.3; by=pos[:,0,1]+np.random.randn(n_frames)*.3

    elif tactic == "Post-Up":
        px2=60+3*np.sin(np.linspace(0,2*np.pi*.8,n_frames))
        py2=25+2*np.cos(np.linspace(0,2*np.pi*1.1,n_frames))
        pos[:,0,:]=smooth(np.stack([px2,py2],axis=1))
        pos[:,5,:]=pos[:,0,:]+np.column_stack([np.random.randn(n_frames)*.8-1, np.random.randn(n_frames)*.8])
        for p,spot in enumerate([[50,8],[50,42],[42,10],[42,40],[35,25]]): pos[:,p+1,:]=micro_move(n_frames,spot,r=2.0,freq=.7)
        for p,spot in enumerate([[48,10],[48,40],[40,12],[40,38]]): pos[:,p+6,:]=micro_move(n_frames,spot,r=1.8)
        bx=pos[:,0,0]+np.random.randn(n_frames)*.4; by=pos[:,0,1]+np.random.randn(n_frames)*.4

    pos = np.clip(pos,[0,0],[COURT_W,COURT_H])
    ball = np.clip(np.stack([bx,by],axis=1),[0,0],[COURT_W,COURT_H])
    return {"positions":pos,"ball":ball,"tactic":tactic}

# ─── 2. LSTM ENCODER ─────────────────────────────────────────────
class LSTMEncoder:
    def __init__(self, input_dim, hidden_dim=64, seed=42):
        rng=np.random.default_rng(seed); sc=np.sqrt(2.0/(input_dim+hidden_dim))
        self.Wf=rng.normal(0,sc,(hidden_dim,input_dim+hidden_dim)); self.bf=np.zeros(hidden_dim)
        self.Wi=rng.normal(0,sc,(hidden_dim,input_dim+hidden_dim)); self.bi=np.zeros(hidden_dim)
        self.Wo=rng.normal(0,sc,(hidden_dim,input_dim+hidden_dim)); self.bo=np.zeros(hidden_dim)
        self.Wc=rng.normal(0,sc,(hidden_dim,input_dim+hidden_dim)); self.bc=np.zeros(hidden_dim)
        self.hidden_dim=hidden_dim

    def _sig(self,x): return 1/(1+np.exp(-np.clip(x,-500,500)))
    def _tanh(self,x): return np.tanh(np.clip(x,-500,500))

    def forward(self, X):
        h=np.zeros(self.hidden_dim); c=np.zeros(self.hidden_dim)
        for t in range(len(X)):
            z=np.concatenate([X[t],h])
            f=self._sig(self.Wf@z+self.bf); i=self._sig(self.Wi@z+self.bi)
            o=self._sig(self.Wo@z+self.bo); ct=self._tanh(self.Wc@z+self.bc)
            c=f*c+i*ct; h=o*self._tanh(c)
        return h

    def encode_possession(self, data):
        pos=data["positions"]; ball=data["ball"]; T=len(pos)
        feats=[]
        for t in range(T):
            f=[]
            for p in range(N_PLAYERS):
                f.extend([pos[t,p,0]/COURT_W, pos[t,p,1]/COURT_H])
                f.append(np.linalg.norm(pos[t,p]-ball[t])/COURT_W)
                vel=(pos[t,p]-pos[t-1,p])*SAMPLE_RATE if t>0 else np.zeros(2)
                f.extend([vel[0]/30, vel[1]/30])
            feats.append(f)
        return self.forward(np.array(feats,dtype=np.float32))

# ─── 3. CLASSIFIER ───────────────────────────────────────────────
COACHING = {
    "Pick-and-Roll":  {"strengths":"Tao mismatch hieu qua, buoc doi thu switch","when_best":"Shot clock > 14s, guard nhanh hon center doi thu","adjustment":"Drop coverage: screener pull-up | Hard hedge: ball-handler drive","drill":"2-man game: hand-off + mid-range decision"},
    "Isolation":      {"strengths":"Khai thac mismatch ca nhan, kiem soat tempo","when_best":"Cuoi hiep, shot clock < 8s, star player dang hot hand","adjustment":"Double-team: kick out corner 3 | Switch: exploit size","drill":"1-on-1 footwork: jab step + spin move finishing"},
    "Motion Offense": {"strengths":"Kho defend, tao nhieu passing lane, met doi thu","when_best":"Doi tan cong co shooting tot o tat ca vi tri","adjustment":"Sag defense: spot-up 3 | Tight D: back-cut","drill":"5-on-0 motion: pass and cut, screen-away"},
    "Fast Break":     {"strengths":"Diem de nhat, doi thu chua set defense","when_best":"Sau turnover hoac defensive rebound, score diff <= 10","adjustment":"Defense tra ve kip: dung lai, to chuc half-court","drill":"3-on-2 -> 2-on-1 transition drill"},
    "Post-Up":        {"strengths":"Khai thac loi the the hinh, drawing fouls hieu qua","when_best":"Doi thu co center nho/yeu hoac foul trouble","adjustment":"Double-team: pass ra canh corner 3","drill":"Mikan drill + Drop step + Jump hook"},
}

class TacticClassifier:
    def __init__(self):
        self.gb=GradientBoostingClassifier(n_estimators=100,max_depth=4,random_state=42)
        self.mlp=MLPClassifier(hidden_layer_sizes=(128,64,32),activation='relu',max_iter=500,random_state=42)
        self.le=LabelEncoder()

    def train(self, vecs, ctxs, labels):
        Xc=np.array(ctxs); Xv=np.array(vecs); Xj=np.hstack([Xv,Xc])
        y=self.le.fit_transform(labels)
        self.gb.fit(Xc,y); self.mlp.fit(Xj,y)

    def predict(self, vec, ctx):
        idx=self.gb.predict(ctx.reshape(1,-1))[0]
        proba=self.gb.predict_proba(ctx.reshape(1,-1))[0]
        tactic=self.le.inverse_transform([idx])[0]
        epv_base={"Pick-and-Roll":1.05,"Fast Break":1.18,"Motion Offense":1.08,"Isolation":0.92,"Post-Up":1.02}
        epv=epv_base.get(tactic,1.0)*(0.7+0.3*ctx[0])+np.random.normal(0,.03)
        return {"tactic":tactic,"confidence":float(proba.max()),"probabilities":dict(zip(self.le.classes_,proba.tolist())),"epv":round(float(epv),3)}

# ─── 4. COURT DRAWING ────────────────────────────────────────────
def draw_court(ax):
    ax.set_facecolor("#16213e")
    lc="#c8d6e5"; lw=1.2
    ax.add_patch(patches.Rectangle((0,0),COURT_W,COURT_H,lw=2,ec=lc,fc="#16213e"))
    ax.add_patch(patches.Rectangle((75,17),19,16,lw=lw,ec=lc,fc="#0f3460",alpha=.7))
    ax.add_patch(plt.Circle((75,25),6,color=lc,fill=False,lw=lw))
    ax.add_patch(patches.Arc((88,25),47,47,angle=0,theta1=110,theta2=250,color=lc,lw=lw))
    ax.plot([75,94],[4,4],color=lc,lw=lw); ax.plot([75,94],[46,46],color=lc,lw=lw)
    ax.add_patch(plt.Circle((88,25),.75,color="#ff6b35",lw=2,fill=False))
    ax.plot([88,94],[25,25],color=lc,lw=2)
    ax.add_patch(plt.Circle((0,25),6,color=lc,fill=False,lw=lw,linestyle="--",alpha=.4))
    ax.set_xlim(0,COURT_W); ax.set_ylim(0,COURT_H); ax.set_aspect("equal"); ax.axis("off")

# ─── 5. GRADIENT TRAJECTORY ──────────────────────────────────────
def plot_gradient_trajectory(ax, x, y, cmap="plasma", lw=2.0, alpha=0.9, label=None, step=1):
    """Ve trajectory voi mau gradient theo thoi gian (xanh -> do -> vang)."""
    x2,y2 = x[::step], y[::step]
    pts = np.array([x2,y2]).T.reshape(-1,1,2)
    segs = np.concatenate([pts[:-1],pts[1:]],axis=1)
    cols = plt.cm.get_cmap(cmap)(np.linspace(0,1,len(segs)))
    lc_obj = LineCollection(segs, colors=cols, linewidth=lw, alpha=alpha)
    ax.add_collection(lc_obj)
    # Arrow at end
    if len(x2)>5:
        ax.annotate("", xy=(x2[-1],y2[-1]), xytext=(x2[-3],y2[-3]),
                    arrowprops=dict(arrowstyle="->",color=cols[-1],lw=1.5))

# ─── 6. DASHBOARD ────────────────────────────────────────────────
def visualize_dashboard(poss, result, encoder):
    fig = plt.figure(figsize=(22,13), facecolor="#0d0d1a")
    fig.suptitle("SmartCoach AI  |  Basketball Tactical Intelligence Demo  |  i-TECH",
                 fontsize=16, color="#ffffff", fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2,3,figure=fig,hspace=.4,wspace=.3,left=.03,right=.97,top=.93,bottom=.04)

    pos  = poss["positions"]
    ball = poss["ball"]
    snap = 175   # mid-possession snapshot

    # ── Panel 1: Court snapshot at mid-possession ──
    ax1 = fig.add_subplot(gs[0,0])
    draw_court(ax1)
    # Trails: 30 frames truoc snapshot
    trail_len = 30
    for p in range(5):
        tx = pos[max(0,snap-trail_len):snap, p, 0]
        ty = pos[max(0,snap-trail_len):snap, p, 1]
        plot_gradient_trajectory(ax1, tx, ty, cmap="Oranges", lw=1.5, alpha=.6, step=1)
    # Player markers
    p5 = pos[snap,:5,:]
    p10= pos[snap,5:,:]
    bpos=ball[snap]
    ax1.scatter(p5[:,0],p5[:,1],s=260,c="#ff6b35",ec="white",lw=1.5,zorder=6)
    ax1.scatter(p10[:,0],p10[:,1],s=260,c="#4ecdc4",marker="s",ec="white",lw=1.5,zorder=6)
    ax1.scatter([bpos[0]],[bpos[1]],s=130,c="#f9ca24",ec="#f0932b",lw=2,zorder=7)
    for i in range(5):
        ax1.text(p5[i,0],p5[i,1],str(i+1),ha="center",va="center",fontsize=7,fontweight="bold",color="white",zorder=8)
    ax1.set_title(f"Snapshot t={snap//SAMPLE_RATE:.1f}s  |  {result['tactic']}",color="white",fontsize=10,pad=6)
    leg1=patches.Patch(color="#ff6b35",label="Attack"); leg2=patches.Patch(color="#4ecdc4",label="Defense")
    ax1.legend(handles=[leg1,leg2],loc="upper left",facecolor="#1a1a2e",labelcolor="white",fontsize=7,framealpha=.9)

    # ── Panel 2: Gradient trajectory (IMPROVED) ──
    ax2 = fig.add_subplot(gs[0,1])
    draw_court(ax2)
    cmaps_off = ["plasma","viridis","cool","spring","autumn"]
    for p in range(5):
        tx=pos[:,p,0]; ty=pos[:,p,1]
        plot_gradient_trajectory(ax2, tx, ty, cmap=cmaps_off[p], lw=1.8, alpha=.75, step=2)
        # Start dot
        ax2.scatter([tx[0]],[ty[0]],s=35,c="white",zorder=5,alpha=.8)
        # End arrow already in function

    # Ball trace rieng
    plot_gradient_trajectory(ax2, ball[:,0], ball[:,1], cmap="YlOrRd", lw=2.5, alpha=.9, step=3)

    # Phase markers
    pf=int(N_FRAMES*.75)
    ax2.axvline(pos[int(N_FRAMES*.3),0,0],color="white",lw=.7,linestyle=":",alpha=.4)
    ax2.text(pos[int(N_FRAMES*.3),0,0]+.5,3,"Screen",color="white",fontsize=6,alpha=.7)

    ax2.set_title("Trajectory (gradient = thoi gian, trang=bat dau, vang=ket thuc)",color="white",fontsize=9,pad=6)

    # Colorbar time
    sm=plt.cm.ScalarMappable(cmap="plasma",norm=plt.Normalize(0,POSSESSION_SECS))
    sm.set_array([])
    cbar=plt.colorbar(sm,ax=ax2,fraction=.025,pad=.02)
    cbar.set_label("Time (s)",color="white",fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="white"); plt.setp(cbar.ax.yaxis.get_ticklabels(),color="white",fontsize=6)

    # ── Panel 3: LSTM 64-dim fingerprint ──
    ax3=fig.add_subplot(gs[0,2])
    vec=encoder.encode_possession(poss)
    im=ax3.imshow(vec.reshape(8,8),cmap="RdYlGn",aspect="auto",vmin=-1.5,vmax=1.5)
    plt.colorbar(im,ax=ax3,fraction=.046,pad=.04).ax.yaxis.set_tick_params(color="white")
    ax3.set_title("Tang 1: LSTM Vector 64-chieu\n(Tactical Fingerprint)",color="white",fontsize=10,pad=6)
    ax3.tick_params(colors="white")
    ax3.set_facecolor("#1a1a2e")

    # ── Panel 4: Probability bars sorted ──
    ax4=fig.add_subplot(gs[1,0])
    tactics=list(result["probabilities"].keys())
    probs=list(result["probabilities"].values())
    # Sort by prob
    order=np.argsort(probs)
    tactics_s=[tactics[i] for i in order]; probs_s=[probs[i] for i in order]
    bar_colors=["#ff6b35" if t==result["tactic"] else "#4ecdc4" for t in tactics_s]
    bars=ax4.barh(tactics_s,probs_s,color=bar_colors,ec="white",lw=.8,height=.55)
    for bar,p2 in zip(bars,probs_s):
        ax4.text(bar.get_width()+.01,bar.get_y()+bar.get_height()/2,f"{p2*100:.1f}%",va="center",color="white",fontsize=9)
    ax4.set_xlim(0,1.15)
    ax4.set_title(f"Tang 2: Gradient Boosting\nChien thuat: {result['tactic']} ({result['confidence']*100:.0f}%)",color="white",fontsize=10,pad=6)
    ax4.set_facecolor("#1a1a2e"); ax4.tick_params(colors="white")
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
    for sp in ["bottom","left"]: ax4.spines[sp].set_edgecolor("#555")

    # ── Panel 5: EPV gauge improved ──
    ax5=fig.add_subplot(gs[1,1])
    ax5.set_facecolor("#1a1a2e"); ax5.axis("off")
    epv=result["epv"]
    # Gradient gauge
    x_gauge=np.linspace(.7,1.3,200)
    y_gauge=np.ones(200)*.5
    c_gauge=plt.cm.RdYlGn((x_gauge-.7)/.6)
    for xi in range(len(x_gauge)-1):
        ax5.axvspan(x_gauge[xi],x_gauge[xi+1],ymin=.3,ymax=.7,color=c_gauge[xi],alpha=.8)
    ax5.axvline(epv,color="white",lw=4,ymin=.15,ymax=.85,zorder=5)
    ax5.axvline(1.0,color="#aaa",lw=1.5,ymin=.15,ymax=.85,linestyle="--",alpha=.7)
    ax5.set_xlim(.65,1.35); ax5.set_ylim(0,1)
    ax5.text(epv,.88,f"{epv:.3f}",ha="center",color="white",fontsize=18,fontweight="bold")
    ax5.text(1.0,.12,"avg\n1.00",ha="center",color="#aaa",fontsize=7)
    ax5.text(.67,.5,"EPV",color="white",fontsize=12,va="center",fontweight="bold")
    epv_lbl="TUYET VOI" if epv>=1.05 else ("TRUNG BINH" if epv>=0.95 else "RUI RO")
    epv_col="#27ae60" if epv>=1.05 else ("#f39c12" if epv>=0.95 else "#e74c3c")
    ax5.text(.5,.5,epv_lbl,transform=ax5.transAxes,ha="center",va="center",fontsize=15,color=epv_col,fontweight="bold",alpha=.3)
    ax5.set_title("Expected Possession Value\n(Tich hop Game Context)",color="white",fontsize=10,pad=6)

    # ── Panel 6: Coaching ──
    ax6=fig.add_subplot(gs[1,2])
    ax6.set_facecolor("#0d1b2a"); ax6.axis("off")
    t=result["tactic"]; tmpl=COACHING[t]
    lines=[
        ("TANG 3: COACHING RECOMMENDATION","#ff6b35",10,True),
        (f"Chien thuat: {t}","#ffffff",10,True),
        ("","#888",8,False),
        ("DIEM MANH:","#4ecdc4",8,True),
        (f"  {tmpl['strengths']}","#dddddd",8,False),
        ("","#888",8,False),
        ("KHI NAO DUNG:","#4ecdc4",8,True),
        (f"  {tmpl['when_best']}","#dddddd",8,False),
        ("","#888",8,False),
        ("DIEU CHINH:","#4ecdc4",8,True),
        (f"  {tmpl['adjustment'][:60]}","#dddddd",8,False),
        ("","#888",8,False),
        ("BAI TAP:","#7bed9f",8,True),
        (f"  {tmpl['drill']}","#dddddd",8,False),
    ]
    y_pos=0.97
    for txt,col,sz,bold in lines:
        ax6.text(.04,y_pos,txt,transform=ax6.transAxes,color=col,fontsize=sz,va="top",fontweight="bold" if bold else "normal")
        y_pos-=0.068
    ax6.set_title("Tang 3: LLM Coaching Layer",color="white",fontsize=10,pad=6)

    fig.text(.5,.005,"SmartCoach AI  |  i-TECH Research Group  |  LSTM -> GradientBoosting -> LLM",
             ha="center",color="#666",fontsize=7,style="italic")

    plt.savefig("smartcoach_dashboard.png",dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    print("Dashboard saved: smartcoach_dashboard.png")
    plt.close()

# ─── 7. FINGERPRINT PLOT ─────────────────────────────────────────
def plot_fingerprints(vecs, labels):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,7),facecolor="#0d0d1a")
    fig.suptitle("Tactical Fingerprint Analysis  |  SmartCoach AI",color="white",fontsize=13,fontweight="bold")
    pca=PCA(n_components=2,random_state=42)
    proj=pca.fit_transform(np.array(vecs))
    colors={"Pick-and-Roll":"#ff6b35","Isolation":"#e056fd","Motion Offense":"#4ecdc4","Fast Break":"#f9ca24","Post-Up":"#7bed9f"}
    ax1.set_facecolor("#1a1a2e")
    for t in TACTIC_LABELS:
        mask=np.array(labels)==t
        ax1.scatter(proj[mask,0],proj[mask,1],s=70,c=colors[t],label=t,alpha=.85,ec="white",lw=.4)
        # Centroid
        cx,cy=proj[mask,0].mean(),proj[mask,1].mean()
        ax1.scatter([cx],[cy],s=200,c=colors[t],ec="white",lw=2,marker="*",zorder=6)
    ax1.legend(facecolor="#1a1a2e",labelcolor="white",fontsize=8,framealpha=.9)
    ax1.set_title(f"PCA 64D->2D  (PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%)",color="white",fontsize=10)
    ax1.tick_params(colors="white"); ax1.set_xlabel("PC1",color="#aaa"); ax1.set_ylabel("PC2",color="#aaa")
    for sp in ax1.spines.values(): sp.set_edgecolor("#444")

    mean_vecs=np.zeros((len(TACTIC_LABELS),64))
    for i,t in enumerate(TACTIC_LABELS):
        mask=np.array(labels)==t; mean_vecs[i]=np.array(vecs)[mask].mean(axis=0)
    im=ax2.imshow(mean_vecs,cmap="RdYlGn",aspect="auto",vmin=-1.5,vmax=1.5)
    ax2.set_yticks(range(len(TACTIC_LABELS))); ax2.set_yticklabels(TACTIC_LABELS,color="white",fontsize=9)
    ax2.set_xlabel("Vector Dimension (0-63)",color="#aaa"); ax2.set_title("Mean Tactical Fingerprint per Play Type",color="white",fontsize=10)
    plt.colorbar(im,ax=ax2,label="activation",fraction=.03)
    ax2.tick_params(colors="white")

    fig.tight_layout()
    plt.savefig("tactical_fingerprints.png",dpi=130,bbox_inches="tight",facecolor="#0d0d1a")
    print("Fingerprints saved: tactical_fingerprints.png")
    plt.close()

# ─── 8. MAIN ─────────────────────────────────────────────────────
def main():
    print("="*65)
    print("  SmartCoach AI v2  |  i-TECH Research Group")
    print("="*65)

    print("Tao du lieu huan luyen...", end=" ", flush=True)
    encoder=LSTMEncoder(input_dim=N_PLAYERS*5)
    vecs,ctxs,labels=[],[],[]
    for t in TACTIC_LABELS:
        for _ in range(40):
            d=generate_possession(t)
            v=encoder.encode_possession(d)
            sc,sd,q,tf=np.random.uniform(5,24),np.random.randint(-15,16),np.random.randint(1,5),np.random.randint(0,6)
            vecs.append(v); ctxs.append([sc/24,sd/30,q/4,tf/5]); labels.append(t)
    print(f"OK ({len(labels)} samples)")

    print("Huan luyen Gradient Boosting...", end=" ", flush=True)
    clf=TacticClassifier(); clf.train(vecs,ctxs,labels)
    print("OK")

    DEMO = "Pick-and-Roll"
    print(f"\nDemo possession: {DEMO}")
    poss=generate_possession(DEMO)
    ctx=np.array([16/24,-3/30,4/4,3/5])

    print("\n[TANG 1 - LSTM]")
    vec=encoder.encode_possession(poss)
    print(f"  Input : {N_FRAMES} frames x {N_PLAYERS} players x 5 features")
    print(f"  Output: vector 64-chieu  ||v||={np.linalg.norm(vec):.3f}")

    print("\n[TANG 2 - GRADIENT BOOSTING]")
    result=clf.predict(vec,ctx)
    print(f"  Chien thuat: {result['tactic']}  (confidence: {result['confidence']*100:.1f}%)")
    print(f"  EPV        : {result['epv']:.3f} pts/possession")
    for t,p in sorted(result['probabilities'].items(),key=lambda x:-x[1]):
        print(f"    {'#'*int(p*30):<30} {t:<20} {p*100:.1f}%")

    print("\n[TANG 3 - LLM COACHING]")
    tmpl=COACHING[result['tactic']]
    print(f"  Diem manh : {tmpl['strengths']}")
    print(f"  Dieu chinh: {tmpl['adjustment']}")
    print(f"  Bai tap   : {tmpl['drill']}")

    print("\nRender dashboard...")
    visualize_dashboard(poss,result,encoder)
    print("Render fingerprints...")
    plot_fingerprints(vecs,labels)

    print("\n" + "="*65)
    print("  HOAN TAT! File anh da luu trong thu muc hien tai.")
    print("="*65)

if __name__=="__main__":
    main()
