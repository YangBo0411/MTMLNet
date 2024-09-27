# MTMLNet: Multi-task Mutual Learning Network for Infrared Small Target Detection and Segmentation
## Usage
### Requirements
### Dataset
In data\custom.yaml, change the path of the corresponding detection and segmentation label dataset.
![image](https://github.com/YangBo0411/MTMLNet/blob/main/fig.jpg)
## The overall architecture
![image](https://github.com/YangBo0411/MTMLNet/blob/main/fig2.png)
## Visualization results
![image](https://github.com/YangBo0411/MTMLNet/blob/main/fig6.png)
## results
<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0
 style='border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:42.35pt'>
  <td valign=top style='border:solid windowtext 1.0pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:42.35pt'>
  <p class=MsoNormal><span class=SpellE><span lang=EN-US style='font-size:10.5pt;
  line-height:150%'>Metheds</span></span><span lang=EN-US style='font-size:
  10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border:solid windowtext 1.0pt;border-left:none;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:42.35pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>R(target)<o:p></o:p></span></p>
  </td>
  <td valign=top style='border:solid windowtext 1.0pt;border-left:none;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:42.35pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>P(target)<o:p></o:p></span></p>
  </td>
  <td valign=top style='border:solid windowtext 1.0pt;border-left:none;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:42.35pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>F1(target)<o:p></o:p></span></p>
  </td>
  <td valign=top style='border:solid windowtext 1.0pt;border-left:none;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:42.35pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>AP<o:p></o:p></span></p>
  </td>
  <td valign=top style='border:solid windowtext 1.0pt;border-left:none;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:42.35pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>R(pixel)<o:p></o:p></span></p>
  </td>
  <td valign=top style='border:solid windowtext 1.0pt;border-left:none;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:42.35pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>P(pixel)<o:p></o:p></span></p>
  </td>
  <td valign=top style='border:solid windowtext 1.0pt;border-left:none;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:42.35pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>F1(pixel)<o:p></o:p></span></p>
  </td>
  <td valign=top style='border:solid windowtext 1.0pt;border-left:none;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:42.35pt'>
  <p class=MsoNormal><span class=SpellE><span lang=EN-US style='font-size:10.5pt;
  line-height:150%'>IoU</span></span><span lang=EN-US style='font-size:10.5pt;
  line-height:150%'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;height:23.0pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:23.0pt'>
  <p class=MsoNormal><span class=SpellE><span lang=EN-US style='font-size:10.5pt;
  line-height:150%'>MDvsMA</span></span><span lang=EN-US style='font-size:10.5pt;
  line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:23.0pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:23.0pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:23.0pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:23.0pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:23.0pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.383 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:23.0pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.435 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:23.0pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.407 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:23.0pt'>
  <p class=MsoNormal><span style='font-size:10.5pt;line-height:150%;font-family:
  等线;mso-ascii-font-family:"Times New Roman";mso-hansi-font-family:"Times New Roman";
  mso-bidi-font-family:"Times New Roman"'>　</span><span lang=EN-US
  style='font-size:10.5pt;line-height:150%;mso-fareast-font-family:等线;
  mso-bidi-font-family:"Times New Roman";color:black'>/</span><span lang=EN-US
  style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2;height:21.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>ACM<o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.735 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.806 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.769 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.627 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;height:21.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>ALC<o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.540 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.860 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.663 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.639 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;height:21.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span class=SpellE><span lang=EN-US style='font-size:10.5pt;
  line-height:150%'>ISNet</span></span><span lang=EN-US style='font-size:10.5pt;
  line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.802 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.695 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.744 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.625 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;height:21.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span class=SpellE><span lang=EN-US style='font-size:10.5pt;
  line-height:150%'>DNANet</span></span><span lang=EN-US style='font-size:10.5pt;
  line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.784 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.801 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.793 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.656 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:20.5pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:20.5pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>UIU<o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:20.5pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:20.5pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:20.5pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:20.5pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:20.5pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.832 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:20.5pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.214 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:20.5pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.340 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:20.5pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman"'>0.612 </span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7;height:21.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  color:red'>Ours<o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:red'>0.819
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:red'>0.826
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:red'>0.822
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:red'>0.692
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:8;height:21.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>OSCAR<o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.620
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:9;height:21.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>EFL<o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.747
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.775
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.761
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.767
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:10;height:21.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%'>MSML<o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.773
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.423
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.547
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.752
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:11;height:21.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span class=SpellE><span lang=EN-US style='font-size:10.5pt;
  line-height:150%'>SSTNet</span></span><span lang=EN-US style='font-size:10.5pt;
  line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.355
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.581
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.441
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>0.659
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:12;mso-yfti-lastrow:yes;height:21.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  color:red'>Ours<o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:red'>0.818
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:red'>0.747
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:red'>0.781
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:red'>0.814
  </span><span lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.15pt'>
  <p class=MsoNormal><span lang=EN-US style='font-size:10.5pt;line-height:150%;
  mso-fareast-font-family:等线;mso-bidi-font-family:"Times New Roman";color:black'>/</span><span
  lang=EN-US style='font-size:10.5pt;line-height:150%;color:red'><o:p></o:p></span></p>
  </td>
 </tr>
</table>

