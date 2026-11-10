"""Regenerate tables and a scientific diagnostic figure from released evidence."""
import argparse
import csv
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    evidence=json.loads((root/'results/evidence.json').read_text(encoding='utf-8'))
    rows=[dict(protocol=protocol,**r) for protocol,items in evidence['main'].items() for r in items]
    with (args.output/'main_results.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scale=json.loads((root/'results/diagnostics/gallery_scale.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained',sharey=True)
    for i,protocol in enumerate(('P0','P1')):
        for j,direction in enumerate(('T2I','I2T')):
            ax=axes[i,j]
            for family,color in [('pretrained_blip','#bf814e'),('pretrained_siglip','#427e9b'),('supervised_siglip','#3b876c')]:
                values=sorted([r for r in scale if r['protocol']==protocol and r['direction']==direction and r['k']==1 and r['family']==family],key=lambda r:r['gallery_places'])
                ax.plot([r['gallery_places'] for r in values],[r['mean'] for r in values],marker='o',ms=3,label=family,color=color)
            ax.set(title=f'{protocol} · {direction}',xlabel='Gallery places',ylabel='Recall@1 (%)');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=7,frameon=False)
    fig.savefig(args.output/'gallery_scale.pdf');fig.savefig(args.output/'gallery_scale.png',dpi=220);plt.close(fig)
    print(f'Wrote tables and figures to {args.output}')

if __name__=='__main__':main()
