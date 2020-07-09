Running the code
================

The code is split into two logical components--doing the training, plotting, and outputing. The following sections talk about them. For reference, the training is taken care of by the ``Utilities/MvMaker.py`` file and the plotting and outputting is done by the ``Utilties/MVAPlotter.py``


Training
********

The training is mandatory for a new output directory (the directory needs to have files to make plots from), but after the first training, the ``-t`` flag can be ignored. The main imports to the training are the variables in between the boxes labeled ``User Variables`` in ``runMVA.py``.

+ **usevar** -- The variables used in the actual training
+ **specVar** -- The variables NOT used in the actual training, but needed (say for cuts)
+ **CUT** -- ROOT style cut string applied as a preselection before training
+ **INPUT_TREE** -- input root for training. Unless you know how to produce this, don't change
+ **groups** -- List of the different training groups with the samples in those training groups

Besides those variables, most of the rest is automated in the training block of the code. The only other change is training in multiclass or binary. The two functions are ``train()`` and ``train_single(sample)``. I would suggest just using the ``train()`` function.

Any other expert changes can be made in the ``./Utilities/MvaMaker.py`` file.

Plotting
********

For plotting, there is plenty of roam for creating one's own graphs, the functions listed here are merely a useful set of functions for making plots. The basic setup for all the functions is as so:

.. code:: python

   output.<func>(<primary>, <other>, <var>, <binning>, <extra name>)

+ **primary group** -- Single group to plot first
+ **other groups** -- Single or collection of groups to plot against
+ **var** -- Variable to plot (BDT variables use the convention ``BDT.<Group>``)
+ **binning** -- binning in the form of a numpy array. Use ``np.linspace`` for this
+ **extra name** -- Option to add a tag at the end of the file name to distinguish it, especially if before and after applying some cut

There is a "show" option that can be turned on in the command line, this makes the matplotlib show the plots before saving them. By default it is turned off.

All of the plots are saved into the output directory. Some of the functions and how they are used are in ``runMVA.py``, but for more functions/information, please look at ``Utilities/MVAPlotter.py`` or the information here.

Output
******

While the code allows for plotting, to plug the BDTs back into the main code base for further analysis, such as running combine on it, it needs to be converted to ROOT format. ``write_out()`` is what is used for writing the pandas information as ROOT histograms. By default, all the histograms are saved with 1024 (unless it has integer argument) to allow for maximum flexibility with rebinning. The format it is written out is for the ``VVAnalysis`` code base, so for more info on that, go to that code base.

If you want to write the code out AFTER some selection, you can apply a cut to the DataFrame stored in the MVAPlotter object using the ``apply_cut(cut)`` function. Also, there is the ``apply_cut_max_bdt(sample)`` function which only keeps events where the BDT value for the sample given is the maximum. After applying a cut to the ``MVAPlotter`` object, the plots will also change, allowing for plotting before and after cut.

Further, if you want to create you're own composite variable, you can grab the DataFrame using ``get_sample()``, create a variable, and add it using ``add_variable(name, arr)``. When you write out to a ROOT file, this variable will included automatically. It will not be included in the pandas DataFrame created by the ``MvaMaker`` object, to save it, one needs to do this manually (though it is not suggested!)

