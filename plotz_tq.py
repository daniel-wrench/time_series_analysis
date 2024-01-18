#!/usr/bin/env python3
import os
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import TurbAn.Analysis.Simulations.AnalysisFunctions as af

from pylab import rcParams
rcParams['figure.figsize'] = 13,8.5

def plotz(tfiles,rfiles,name,plotpoints):
   pdf=PdfPages(TRB_DIR+'e1_8hr_'+name+'_tq.pdf')
   for flt,flr in zip(tfiles,rfiles):
      print(tfiles.index(flt),flt)
      d=pickle.load(open(flt,'rb'))
      dr=pd.read_pickle(flr)
      jumps=len(dr)//plotpoints

      if name in ['b','vp','zpp','zmp','b_v']:
         names=[name+'r',name+'t',name+'n']
      else:
         names=[name]
#     try:
      plt.clf() 
      ax0=plt.subplot2grid((2,4),(0,0),colspan=4)
      ax1=plt.subplot2grid((2,4),(1,0))
      ax2=plt.subplot2grid((2,4),(1,1))
      ax3=plt.subplot2grid((2,4),(1,2))
      ax4=plt.subplot2grid((2,4),(1,3))
      
      if len(names) == 3:
        ax0.plot(dr.index[::jumps],dr[names[0]][::jumps],'.-',label=names[0])
        ax0.plot(dr.index[::jumps],dr[names[1]][::jumps],'.-',label=names[1])
        ax0.plot(dr.index[::jumps],dr[names[2]][::jumps],'.-',label=names[2])
      else:
        ax0.plot(dr.index[::jumps],dr[names[0]][::jumps],'.-',label=names[0])
      ax0.set_xlabel('date')
      ax0.set_title('Bin t='+str(dr.index[len(dr.index)//2]))
      ax0.legend()

      if len(names) == 3:
         ax1.plot(d[name]['times'],d[name]['cr']/3.,label='Cr')
         tcor=d[name]['times'][np.argmin(np.abs(d[name]['cr']/3.-1./np.e))]
      else:
         ax1.plot(d[name]['times'],d[name]['cr'],label='Cr')
         tcor=d[name]['times'][np.argmin(np.abs(d[name]['cr']-1./np.e))]
      ax1.axvline(tcor,linestyle='--',color='r')
      ax1.text(tcor*1.1, 0.8,r'$t_{{corr}}$ = {0:.02f} s'.format(tcor))
      ax1.set_xlabel('$\Delta$ t (s)')
      ax1.legend()

      ax2.loglog(d[name]['times'],d[name]['sfn'][0,:],label='S$^{(2)}$')
      yyy=d[name]['sfn'][0,:][58]
      af.pltpwrl(50,yyy*2.0,xi=1,xf=500,alpha=2./3,label='2/3',ax=ax2)
      yy2=d[name]['sfn'][0,:][1]*1.1
      ax2.text(3,yy2,r'S$^{{(2)}}(dt_{{max}})$={0:.03e}'.format(d[name]['sfn'][0,-1]))
      ax2.set_xlabel('$\Delta$ t (s)')
      ax2.legend()

      if len(names) == 3:
        ax3.semilogx(d[names[0]]['tau'],d[names[0]]['sdk'],label='$\kappa_{'+names[0]+'}$')
        ax3.semilogx(d[names[1]]['tau'],d[names[1]]['sdk'],label='$\kappa_{'+names[1]+'}$')
        ax3.semilogx(d[names[2]]['tau'],d[names[2]]['sdk'],label='$\kappa_{'+names[2]+'}$')
      else:
        ax3.semilogx(d[name]['tau'],d[name]['sdk'],label='$\kappa_{'+names[0]+'}$')
      ax3.set_xlabel('$\Delta$ t (s)')
      ax3.legend()

      if len(names) == 3:
         ax4.semilogy(d[names[0]]['bn1'   ],d[names[0]]['pdf1'   ],label=r'$\Delta = 1 dt$')
         ax4.semilogy(d[names[0]]['bn10'  ],d[names[0]]['pdf10'  ],label=r'$\Delta = 10 dt$')
         ax4.semilogy(d[names[0]]['bn100' ],d[names[0]]['pdf100' ],label=r'$\Delta = 100 dt$')
         ax4.semilogy(d[names[0]]['bn1000'],d[names[0]]['pdf1000'],label=r'$\Delta = 1000 dt$')
      else:
         ax4.semilogy(d[name]['bn1'   ],d[name]['pdf1'   ],label=r'$\Delta = 1 dt$')
         ax4.semilogy(d[name]['bn10'  ],d[name]['pdf10'  ],label=r'$\Delta = 10 dt$')
         ax4.semilogy(d[name]['bn100' ],d[name]['pdf100' ],label=r'$\Delta = 100 dt$')
         ax4.semilogy(d[name]['bn1000'],d[name]['pdf1000'],label=r'$\Delta = 1000 dt$')
      ax4.legend()
      ax4.set_xlabel(r'$\Delta$'+names[0]+r'$/\sigma_{\Delta '+names[0]+'}$')
      ax4.set_ylabel('PDF')
      
      plt.tight_layout()
      pdf.savefig(bbox_inches='tight')
      plt.close()
#     except:
#        pass
   pdf.close()

plotpoints=400
basedir=os.path.abspath(input('Base dir? '))
RAW_DIR=basedir+   '/8hr_chunks/'
TRB_DIR=basedir+'/tq_8hr_chunks/'
tfiles=sorted(glob.glob(TRB_DIR+'e1_8hr_*.p'))
rfiles=sorted(glob.glob(TRB_DIR+'df_e1_8hr_*.p'))

allnames=['vp','b','np_moment','np','wp_fit','va','b_v','zpp','zmp','sc']
#allnames=['np_moment','np','wp_fit','va','sc']
for name in allnames:
   plotz(tfiles,rfiles,name,plotpoints)