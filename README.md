## Multi-Object Tracking with Correlation Filters ##
This is just a wrapper of sinle object tracker [DCFNet](https://github.com/foolwood/DCFNet)


##Start##
---

` git clone https://github.com/ZongweiZhou1/MOT_Correlation_Filter.git `

## Requirements ##
---
Requirements for [MatConvNet](http://www.vlfeat.org/matconvnet/install/)
 - Download MatConvNet 
   ` cd <MOT_CF> `
   ` git clone https://github.com/vlfeat/matconvnet.git `
 - Compile MatConvNet
   Run following command in MATLAB
     ` cd matconvnet  `
     ` run matlab/vl_compilenn `
     Note: Maybe some issuses encounted here, please refer to [MatConvNet](http://www.vlfeat.org/matconvnet/install/) for help.
     
## Tracking ##
---
Here, we just test the [MOTChallenge dataset](https://www.baidu.com/link?url=iW_UMylrB9lui1bCCZ2lxI0BsUldG3H_jtX7LdNTNWqJITHt73G_8YxaMsbsA7lZ&wd=&eqid=db69568e000027d1000000065a12a0f1), and the parameters can be further adjusted.
