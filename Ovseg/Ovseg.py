import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

from slicer.util import setSliceViewerLayers
import numpy as np
import SimpleITK as sitk
import sitkUtils

import tempfile

#
# Ovseg
#

class Ovseg(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Ovarian cancer CT Segmentation" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Segmentation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Paolo Zaffino (Magna Graecia University of Catanzaro, Italy)", "Thomas Buddenkotte (University Medical Center HamburgEppendorf, Germany)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = '''
This module segments ovarian cancer tissues in CT images.
Ovseg library is described ''' + f'<p> <a href="{"https://github.com/ThomasBudd/ovseg/tree/main"}">here</a>.</p>'
    self.parent.helpText += f'<p>For more information see the <a href="{"https://github.com/ThomasBudd/ovseg/tree/main"}">Ovseg website</a>.</p>'
    self.parent.acknowledgementText = """ """ # replace with organization, grant and thanks.

#
# OvsegWidget
#

class OvsegWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """


  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # CT volume selector
    #
    self.CTSelector = slicer.qMRMLNodeComboBox()
    self.CTSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.CTSelector.selectNodeUponCreation = True
    self.CTSelector.addEnabled = False
    self.CTSelector.removeEnabled = False
    self.CTSelector.noneEnabled = False
    self.CTSelector.showHidden = False
    self.CTSelector.showChildNodeTypes = False
    self.CTSelector.setMRMLScene(slicer.mrmlScene)
    self.CTSelector.setToolTip( "Select the CT" )
    parametersFormLayout.addRow("CT volume: ", self.CTSelector)

    #
    # output volume selector
    #

    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.outputSelector.selectNodeUponCreation = True
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    self.outputSelector.noneEnabled = True
    self.outputSelector.showHidden = False
    self.outputSelector.showChildNodeTypes = False
    self.outputSelector.setMRMLScene(slicer.mrmlScene)
    self.outputSelector.baseName = "Ovarian cancer segmentation"
    self.outputSelector.setToolTip("Select or create a segmentation for ovarian cancer tissues")
    parametersFormLayout.addRow("Output segmentation: ", self.outputSelector)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply (it can take some minutes)")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.CTSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

    # Create logic object
    self.logic = OvsegLogic()

  def onSelect(self):
    self.applyButton.enabled = self.CTSelector.currentNode() and self.outputSelector.currentNode()


  def onApplyButton(self):
    self.logic.run(self.CTSelector.currentNode(), self.outputSelector.currentNode())
#
# OvsegLogic
#

class OvsegLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def run(self, CTVolume, outputSegmentation):
    """
    Run segmentation
    """

    # Import the required libraries
    try:
      from ovseg.run.run_inference import run_inference
    except ModuleNotFoundError:
      print("Ovseg don't found. It will be installed. It can take time.")
      slicer.util.pip_install("https://github.com/ThomasBudd/ovseg/archive/refs/heads/main.zip")
      from ovseg.run.run_inference import run_inference

    # Create temporary directory and save CT as nii.gz
    tempDir = tempfile.TemporaryDirectory()
    CT_sitk = sitk.Cast(sitkUtils.PullVolumeFromSlicer(CTVolume.GetName()), sitk.sitkFloat32)
    CT_temp_name = tempDir.name + os.sep + "CT.nii.gz"
    sitk.WriteImage(CT_sitk, CT_temp_name)

    # Run Ovseg inference
    run_inference(CT_temp_name, fast=True)

    # Read the segmentation file and remove temporary direcotry
    outputSegmentation_tempfile = tempDir.name + os.sep + "ovseg_predictions_pod_om" + os.sep + "CT.nii.gz"
    outputSegmentation_sitk=sitk.ReadImage(outputSegmentation_tempfile)

    tempDir.cleanup()

    # Create labelmaps
    label_slicer = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    label_slicer = sitkUtils.PushVolumeToSlicer(outputSegmentation_sitk, label_slicer)

    # Convert labelmaps to segmentations
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label_slicer, outputSegmentation)
    outputSegmentation.CreateClosedSurfaceRepresentation()
    slicer.mrmlScene.RemoveNode(label_slicer)

    # Rename segments
    for i in range(len(outputSegmentation.GetSegmentation().GetSegmentIDs())):
      segmentation_name = outputSegmentation.GetSegmentation().GetNthSegment(i).GetName()

      if segmentation_name == "Label_1":
        outputSegmentation.GetSegmentation().GetNthSegment(i).SetName("Omentum")
      elif segmentation_name == "Label_9":
        outputSegmentation.GetSegmentation().GetNthSegment(i).SetName("Main disease")

