"""build_modelling_artifact.py — render the MODELLING artifact page (Results / Methods / Extended: legends,
definitions and derivations / working appendix) from docs/paper/modelling_draft.md, embedding the Fig. 5
render and the full derivations of the companion note (docs/artifacts/symmetry/derivations.html in ~/rnn).

  cd /home/leon/dual && python3 build_modelling_artifact.py
then publish figures/paper_share/artifact_build/mpfc_dual_modelling_v1.html with the Artifact tool (same URL)."""
import re, subprocess, base64, os
ROOT='/home/leon/dual'; BUILD=f'{ROOT}/figures/paper_share/artifact_build'; MD=f'{ROOT}/docs/paper/modelling_draft.md'
DERIV='/home/leon/rnn/docs/artifacts/symmetry/derivations.html'; FIGDIR=f'{ROOT}/figures/paper_share/modelling'
md=open(MD).read().split('\n')
def b(pred, what):
    h=[i for i,l in enumerate(md) if pred(l)]; assert h, what; return h[0]
i_res=b(lambda l:l.strip()=='<!-- RESULTS -->','RESULTS'); i_met=b(lambda l:l.startswith('## Methods'),'Methods')
i_ext=b(lambda l:l.startswith('## Extended:'),'Extended'); i_app=b(lambda l:l.startswith('## Working appendix'),'appendix')
chunks={'intro':md[:i_res],'results':md[i_res:i_met],'methods':md[i_met:i_ext],'extended':md[i_ext:i_app],'appendix':md[i_app:]}
html={k:subprocess.run(['pandoc','-f','gfm','-t','html','--mathjax'],input='\n'.join(v),capture_output=True,text=True,check=True).stdout for k,v in chunks.items()}
# the derivations: body between the toc and the foot, plus its box/proof CSS
d=open(DERIV).read(); css=re.search(r'<style>(.*?)</style>', d, re.S).group(1)
keep=[ln for ln in css.split('\n') if re.match(r'\s*(\.box|\.proof|\.qed|\.tbl-scroll|table|th, td|thead th|td\.num|\.cayley|mjx-container|h2|h3)', ln)]
body=d[d.index('<h2 id="s0">'):d.index('<p class="foot">')]
html['extended']=html['extended'].replace('<!-- DERIVATIONS -->', '<div class="deriv">'+body+'</div>')
# figures: embed every png in FIGDIR whose stem appears in the legends as "Figure 5" / "Extended Data Fig. N"
def uri(fp):
    """embed a web copy (1800 px wide JPEG) so the page stays well under the 16 MB artifact limit; the 400-dpi PNG/SVG stay on disk"""
    web = fp.replace('.png', '_web.jpg'); subprocess.run(['convert', fp, '-resize', '1800x>', '-quality', '86', web], check=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(open(web,'rb').read()).decode()
fig5=f'{FIGDIR}/fig5_model.png'
if os.path.exists(fig5):
    html['extended']=re.sub(r'(<p>Figure 5 \|)', f'<p><img src="{uri(fig5)}" alt="Figure 5" style="max-width:100%"></p>\n\\1', html['extended'], count=1)
for n in range(11,20):
    fp=f'{FIGDIR}/ed{n}.png'
    if os.path.exists(fp): html['extended']=re.sub(rf'(<p>Extended Data Fig\. {n} \|)', f'<p><img src="{uri(fp)}" alt="ED {n}" style="max-width:100%"></p>\n\\1', html['extended'], count=1)
tpl=open(f'{BUILD}/modelling_template.html').read()
v=re.search(r'draft (v[\d.]+)', md[0]).group(1)
page=tpl.replace('{{RES_V}}',v).replace('{{DIS_V}}','—').replace('{{INTRO}}',html['intro']).replace('{{RESULTS}}',html['results']).replace('{{METHODS}}',html['methods']).replace('{{APPENDIX}}',html['appendix'])
page=page.replace('<div class="part" id="discussion">\n<p class="kicker">Discussion</p>\n{{DISCUSSION}}\n</div>', '<div class="part" id="extended">\n<p class="kicker">Extended: figure legends, definitions and derivations</p>\n'+html['extended']+'\n</div>')
page=page.replace('</style>', '\n'.join(keep)+'\n.deriv .box{border-left:3px solid var(--accent);background:var(--note-bg);padding:10px 14px;margin:14px 0;border-radius:0 4px 4px 0}\n.deriv .proof{font-size:14px;color:var(--muted);margin:6px 0 14px}\n.deriv .tag{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--teal);display:block;margin-bottom:6px}\n.deriv table{font-size:14px}\n</style>\n<script>window.MathJax={tex:{inlineMath:[["$","$"],["\\\\(","\\\\)"]],displayMath:[["$$","$$"],["\\\\[","\\\\]"]],macros:{R:"\\\\mathbb{R}",E:"\\\\mathbb{E}",Tr:"\\\\mathrm{T}",diag:"\\\\operatorname{diag}",sgn:"\\\\operatorname{sgn}",relu:"\\\\operatorname{relu}",kap:"\\\\kappa",bk:"\\\\boldsymbol{\\\\kappa}"}},svg:{fontCache:"global"},options:{skipHtmlTags:["script","noscript","style","textarea","pre","code"]}};</script>\n<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js" async></script>',1)
assert not re.findall(r'\{\{[A-Z_]+\}\}', page), re.findall(r'\{\{[A-Z_]+\}\}', page)
out=f'{BUILD}/mpfc_dual_modelling_v1.html'; open(out,'w').write(page); print('built', len(page)//1024, 'KB', v, '| fig5:', os.path.exists(fig5))
