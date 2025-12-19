The work is probably too long to check exhaustively.
The main reason for this is the rather long descriptions of the out-of-distribution
detection methods as well as the search methods. Both together are alone about 20 pages long.
I am not sure, however, how I can shorten this without outright not describing some of the methods.

Other concerns are the topology section,
as I mainly use six to seven definitions from my previous work, plus four new ones simply to define compactness.
This, again, is a concept from topology but relevant here, as this is basically what a bounded space is called for Lie groups.

Similarly, my proof that the decomposition for the affine groups into rotation,shear, etc., is very convoluted.
It would be nice to completely remove it,
but I cannot find a citation that the affine group can be decomposed
into reflection, rotation, shear, scale, and translation,
even though this is a rather standard result.

It would be nice if you could take a look at the ITS section in the analysis
part to check whether it interpreted everything in the Tilt Your Head paper correctly.
The paper seems to describe that hypotheses are across branches,
but the code seems to implement that after the first split, the best hypothesis is selected per branch.

The other important section is likely the last part of the evaluation
that compares against supervised methods on the different datasets,
and especially the part where I compare against ITS on the Vision Transformer and SI score.

It would be interesting how you loaded the ViT16b model,
as my score on SI score was 38.8, not 38.5 like in the ITS paper.
Similarly, the energy reached 51.4, but that could be due to different sampling,etc.
Also, you only mention how you fine-tune the ViT by sampling rotation and scale—on how much data
did you fine-tune the pretrained ImageNet model on, what transformation did ouy apply to si score only normalization or also the standard preprocessing from Vit that uses cropping?

Also, some questions regarding submission: it is stated in the Studienordnung that the thesis
should be printed twice—is that correct?
In addition, do you know if the second supervisor is known yet?

Thanks for your time and help,
Dominik