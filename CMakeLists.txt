cmake_minimum_required(VERSION 3.13.4)

project(Ovseg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/pzaffino/SlicerOvseg")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Paolo Zaffino (Magna Graecia University of Catanzaro, Italy), Thomas Buddenkotte (University Medical Center HamburgEppendorf, Germany)")
set(EXTENSION_DESCRIPTION "This extension segments ovaian cancer tissues in CT images")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/pzaffino/SlicerOvseg/main/Ovseg.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/pzaffino/SlicerOvseg/main/Ovseg_screenshot.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(OvSeg)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
