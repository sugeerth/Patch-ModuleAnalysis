import sys
from pympler.asizeof import asizeof
import operator
import timeit
import time
import pympler
from OpenGL.GL.ARB.vertex_buffer_object import *
import pprint
import pycuda.gl
import pycuda.gl
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import cProfile as profile
import re
import pycuda.gpuarray as gpuarray
import threading
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from pympler import summary
from PySide.QtCore import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from time import sleep
from OpenGL.GLU import *
from OpenGL.GLE import *
from decimal import Decimal
from GLModule import *
from math import sqrt
from boxfish.gl.GLWidget import GLWidget
from PySide import QtGui, QtCore
import TorusIcons
from boxfish.ColorMaps import ColorMap, ColorMapWidget, drawGLColorBar
from PySide.QtGui import QWidget,QLabel,QPixmap,QLineEdit,QHBoxLayout,qRgba,\
    QImage,QVBoxLayout,QComboBox,QCheckBox,QSpacerItem,QIntValidator
from boxfish.gl.GLWidget import GLWidget, set_perspective
from boxfish.gl.glutils import *
import matplotlib.cm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#print rank, size

"""This class acts as an agent to the existing , it like the MODEL for the MODEL VIEW CONTROLLER
Design pattern,"""
class Patch3dAgent(GLAgent):
    """ This is an agent for modules that deal with application patches. """
    #input_level = 0
    patchUpdateSignal = Signal(list, list)
    highlightUpdateSignal = Signal(list)
    PatchSceneUpdateSignal = Signal(ColorMap, tuple)
   
    def __init__(self, parent, datatree):
	super(Patch3dAgent, self).__init__(parent, datatree)
	self.var = 0
	self.addRequest("patches")
	self.patchid = None
	self.values = None
	self.patch_table = None
	self.patch_coords_dict = dict()
	self.coords_patch_dict = dict()
	self.run = None
	if sys.platform == "win32":
		print sys.platform
	else:
		self.platform=sys.platform
	self.requestUpdatedSignal.connect(self.updatePatchValues)
	self.highlightSceneChangeSignal.connect(self.processHighlights)
	self.attributeSceneUpdateSignal.connect(self.processAttributeScenes)	


    def registerPatchAttributes(self, indices):
	#print indices
	self.run_var=self.datatree.getItem(indices[0]).getRun()
	self.registerRun(self.run_var)
	self.requestAddIndices("patches", indices)

    def registerRun(self, run):
	#global level
	if run is not self.run:
	    self.run = run
	    application = run["application"]
	    centers = application["centers"]
	    sizes = application["sizes"]
	    level= application["levels"]
	    Owner = application["Owner"]
	    Hops = application["Hops-Extents"]
	    self.patch_table = run.getTable(application["patch_table"])
	    self.patch_coords_dict, self.coords_patch_dict = \
 		self.patch_table.createIdAttributeMaps(centers + sizes +  level + Owner + Hops)	
	    #print level
	    #print pprint.pprint(self.patch_coords_dict)
    @Slot()
    def processHighlights(self):
	"""When highlights have changed, projects them onto the domains
	   we care about and signals the changed local highlights.
	"""
	if self.run is not None:
	    self.patch_highlights = self.getHighlightIDs(self.patch_table, self.run)
	    self.Ids_from_other_modules=self.patch_highlights
	    self.highlightUpdateSignal.emit(self.patch_highlights)
	    

    def map_node_color(self, val, preempt_range = 0):
	"""Turns a color value in [0,1] into a 4-tuple RGBA color.
	   Used to map nodes.
	"""
	return self.patch_cmap.getColor(val, preempt_range)
 
    def selectionChanged(self,highlight_ids):
	self.highlight_ids=highlight_ids
	tables = list()
	runs = list()
	id_lists = list()
	for id_set in self.highlight_ids:
	    runs.append(self.run)
	    id_lists.append(id_set[1])
	    if id_set[0] == "patches":
		tables.append(self.patch_table)

	#print "This guy is going to be highlighted!  =>  :",self.highlight_ids
	#Talks to the Module and letting it know that the selection has been made and the announces the Change to the BoxFish.
 
	self.setHighlights(tables, runs, id_lists)

     
    @Slot()
    def processAttributeScenes(self):
	patchScene = self.requestScene("patches")
	#print patchScene.color_map,"   \n",patchScene.total_range,"\n"
	self.PatchSceneUpdateSignal.emit(patchScene.color_map, patchScene.total_range)
   
    @Slot(str)
    def updatePatchValues(self):
	self.patchid, self.values = self.requestOnDomain("patches",
	    domain_table = self.patch_table,
	    row_aggregator = "mean", attribute_aggregator = "mean")
	#print self.patch_table	
	self.patchUpdateSignal.emit(self.patchid, self.values)
	#print "Emmitted",self.patchid
	"""This class designs the frame that is to be associated with the widget."""



@Module("3D Patch View", Patch3dAgent, GLModuleScene)
class Patch3dFrame(GLFrame):    
    def __init__(self, parent, parent_frame = None, title = None):
	super(Patch3dFrame, self).__init__(parent, parent_frame, title)
	self.agent.patchUpdateSignal.connect(self.updatePatchData)
	self.agent.highlightUpdateSignal.connect(self.glview.updateHighlights)
	self.droppedDataSignal.connect(self.droppedData)
	self.agent.PatchSceneUpdateSignal.connect(self.glview.updateScene)
	self.color_tab_type = PatchColorTab

    def createView(self):
	widget = QWidget()
	layout = QVBoxLayout()

	self.currentTime = glutGet(GLUT_ELAPSED_TIME);

	grid = QtGui.QGridLayout()
	grid.setSpacing(10)


	self.glview = Patch3dGLWidget(self)  

	self.Slider_for_Spread= QtGui.QSlider(QtCore.Qt.Horizontal, self)
	self.Slider_for_Spread.setFocusPolicy(QtCore.Qt.StrongFocus)	
	self.Slider_for_Spread.setRange(0,15)
	self.Slider_for_Spread.setValue(0)
	self.Slider_for_Spread.valueChanged[int].connect(self.glview.changeValue_for_Spread)


#SLider for patch levels


	self.Slider_in_PatchModule3D = QtGui.QSlider(QtCore.Qt.Horizontal, self)
	self.Slider_in_PatchModule3D.setFocusPolicy(QtCore.Qt.StrongFocus)	
	self.Slider_in_PatchModule3D.setRange(0,2)
	self.Slider_in_PatchModule3D.setValue(0)
	self.Slider_in_PatchModule3D.valueChanged[int].connect(self.glview.changeValue_for_Level_Slider)
	self.agent.requestScene("patches").announceChange()
    

#SLider for Opacity factors

	self.Transp= QtGui.QSlider(QtCore.Qt.Horizontal, self)
	self.Transp.setFocusPolicy(QtCore.Qt.StrongFocus)	
	self.Transp.setRange(0,15)
	self.Transp.setValue(10)
	self.Transp.valueChanged[int].connect(self.glview.changeValue_for_transparancy)
	#self.agent.requestScene("patches").announceChange()

	self.level_label = QtGui.QLabel('Level',self)
	self.Spread_Label = QtGui.QLabel('Spread:',self)

	self.Spread_Label.hide()

	self.Transparent_Label = QtGui.QPushButton('DICE',self)
	self.Transparent_Label.setCheckable(True)
    	self.Transparent_Label.clicked[bool].connect(self.glview.changeTitle)


#Push button for the purpose of magnification 

	self.Neigh_Label = QtGui.QPushButton('MAGNIFY',self)
	self.Neigh_Label.setCheckable(True)
    	self.Neigh_Label.clicked[bool].connect(self.glview.Magnifymode)


#Push button for x, y and z slice planes.


	self.Plane_x = QtGui.QPushButton('X',self)
	self.Plane_x.setCheckable(True)
    	self.Plane_x.clicked[bool].connect(self.glview.changePlane_x)	

	self.Plane_y = QtGui.QPushButton('Y',self)
	self.Plane_y.setCheckable(True)
    	self.Plane_y.clicked[bool].connect(self.glview.changePlane_y)	

	self.Plane_z = QtGui.QPushButton('Z',self)
	self.Plane_z.setCheckable(True)
    	self.Plane_z.clicked[bool].connect(self.glview.changePlane_z)	


	self.Plane_x.hide()
	self.Plane_y.hide()
	self.Plane_z.hide()
    	self.Slider_for_Spread.hide()
	self.Transp.hide()

	grid.addWidget(self.level_label,1,0)
	grid.addWidget(self.Slider_in_PatchModule3D,1,1)
	grid.addWidget(self.Plane_x,1,2)
	grid.addWidget(self.Plane_y,1,3)
	grid.addWidget(self.Plane_z,1,4)
	grid.addWidget(self.Transparent_Label,1,5)
	grid.addWidget(self.Transp,1,6)
	grid.addWidget(self.Neigh_Label,1,7)

	layout.addLayout(grid)
	layout.addWidget(self.glview)
	widget.setLayout(layout)
	return widget

    @Slot(list, str)
    def droppedData(self, indexList):
	self.agent.registerPatchAttributes(indexList)

    @Slot(list, list)	
    def updatePatchData(self, patchid, values):

	self.glview.patchid = patchid
	self.glview.values = values
	self.glview.updateGL()
"""
This class is responsible for rendering of the visualization as well as contains all the widget data that is supposed
to be in a class 
The Widget returns the object by 
Object = Patch3dGLWidget(). 
One could simply call this and get the class to register onto its classes.
"""
"""
Documentaiton
"""

class SortedDisplayDict(dict):
   def __str__(self):
       return "{" + ", ".join("%r: %r" % (key, self[key]) for key in sorted(self))	

"""
SetPatchSize(): (Function responsible for implosion operators)
It sets the size of the patches and announces the change to the patch module. As soon as the patch
sizes are updated the display lists are updated to reflect the change in rendering of patches.

ChangeValue_for_transparency(): (Function to change the opacity of patches)The function sets a value when the Opacity slider is updated.ChangeValue_for_Level_slider(): (Function to change the level of interest)
Visualizes the desired level of interest.

updateHighlights (): 
Given a list of the patch ids to be highlighted, it displays a dialogue box informing the data-attributes 
of the patches of interest. 

Neighbour Change(): (Function that finds the neighbours of the patch of interest.) 
It returns the neighbours of the patch of interest. Does not return the patches which are already 
sliced-off using the Slice and Dice operation.

Process_ planes():(Function that initiates values for slice planes):  
Initiates the planes (Hashmap) to render during the slice and dice mode. 

Magnify (Function that returns patches in the vicinity of the patch of interest.):
Finds the distance between the patch of interest and the patches surrounding the patch of interest.

Range(): 
Calculates the distance between the patch of interest and the entire simulated domain. It cuts them 
into pieces to facilitate the process of placing the slice planes with equal intervals between them.

DoPick(): 
A function responsible for the selection of patches and slice planes. 

Plane_highlight(): 
Function responsible for the selection of a particular plane in the simulated domain. 

Highlight_Drawing(): 
Visualizes:

		    Multiple patch of interest

		   Level of interest

		    Planes for Slice and dice

		    Magnification 

		    Neighbours of the patch 

		    Involves implosions and opacity sliders to distinguish the patch of interest.

DrawCubes(): 
It renders the simulated domain when no plane or patch is selected. 

 ChangeTitle(): 
Hides/shows the Qt widgets to facilitate the magnify operation in the module. 


changePlane_x, changePlane_y, changePlane_z():
Set the class variables to true when the Qt widget x,y and z is selected. 

init():  
It initializes all the class variables. 

createView() in Patch3dFrame:  
Creates Qt widgets in the Patch module.

The patch widget has main functions such as: 
CalculateFPS():Finds the FPS when the MOuse move button function is initiated.This is usually from GLWidget
doLegend():
updateScene():Responsible for updating the displayList and propagating the chage when there is a different data attribute dropped onto the scene.
PlaneUpdate():UPdates the DisplayList, Uses the CubeList to update if there is no highlighted id,Uses HighlightList to update when a region of interest is picked.
Change_plane_x():Sets the planes in the X-axis
Change_plane_y():Sets the plane in y0axis 
Change_plane_z():Sets the plane in Z-axis
ChangeTitle():Is a qt control function where the function is called when there is a change in the QPushButton Dice.Also shows the hidden Qt widgets in the Patch3dFrame Module
ChangeAxis():
drawCubes():Renders the scene by retriving the data from the dictionary.The rendering is sequential and step by step. If this could be parallelized then the rendering associated would immensely help in the optimization of the application 
Seeing ways to use pycuda or pyopengl parallely. The algorithm that we might use is the homogenous weiighted subdivision or it might be 
homogenous subdivision. If the communication between the preocessors needds to be minimized then we would have to see an alsgorithm called binary-swap composition. 
Find_z_Value():Finds the Value of Z of a given patch id.We find that If we could sort the Z-values then the rendering will be more smoother and aweseome. 
highlightDrawing():The function associated with the rendering picking of a patch and re-drawing the whole scene with the highlighted patch, patch-neighbours and the highlighted magnify operation. 
PLanehighlight():
DoPick():
MagnifyMode():
mousepressevent():Initiates the DoPick() function where the selection opeartion can be performed.It also pops out the advanced options in the module.This could be an implosion operation or changing the colormap or usiing the propagation functionality of Boxfish.
SetPatchSie():Used in implosion operation It sets the patch size according to the slider in the Qt widgets. 
PaintGL():
map_node_color();Changes the color of the patches in the simulated domain.
Magnify():Finds the patch ids which give a kind of like magnify effect to the scene.
PatchNeighbours():Finds the neighbors of the patch within the vicinity of the patch. 
KeyreleaseEvent():Uses many keyboard shortcuts to manipulate the data.
DrawProjection Highlights():Used when the projection is made out of the given rendering module.
NeighbourChange():Making new neighbours of the scene 
updateHighlights():Updates the highlights used in the scene to propagate through the other modules of boxfish,
drawPatchColorBar():Draws thecolor bar
changeValue_for_Level_Slider():Changes the level of refinement the current scenen is in,
drawAxis():Draws the axis using a DisplyList.
doAxis():Draws the axis using a DisplayList.
"""

class Patch3dGLWidget(GLWidget):
    
    PatchColorChangeSignal = Signal()
    
    def __init__(self, parent,**keywords):
	#global input_level
	super(Patch3dGLWidget, self).__init__(parent)
	def Template(name, default_value):
	    setattr(self, name, keywords.get(name, default_value))
	self.parent = parent
	self.patch_x=self.patch_y=self.patch_z=0
	self.patch_cmap=self.parent.agent.requestScene("patches").color_map
	#print self.parent.agent.requestScene("patches")
	self.trans=True
	self.Wire_Frame=True
	self.counter=0
	self.highlight_ids = []
	self.number_of_ticks = 10
	self.PlaneValues=[]
	self.initial_size = 512,512
	self.current_sizecdd = self.initial_size
	self.animate = True
	self.enable_cuda = True
	self.window = None     # Number of the glut window.
	self.time_of_last_draw = 0.0
	self.time_of_last_titleupdate = 0.0
	self.frames_per_second = 0.0
	self.frame_counter = 0
	self.output_texture = None # pointer to offscreen render target
	(source_pbo, dest_pbo, cuda_module, invert,
	 pycuda_source_pbo, pycuda_dest_pbo) = [None]*6
	self.heading,self.pitch,self.bank = [0.0]*3
	self.Frame_rate=0
	self.PLane_state=False
	self.propagate=False
	self.neighbours=[]
	self.flag=True
	self.fps=0
	self.enable_cuda=True
	self.currentTime =0
	self.previousTime =0
	self.new=[]
	self.partition = None
	self.Highlight_from_other_Modules=[]
	self.face_renderer = None
	self.solid_faces = []
	#self.solid_face_list = DisplayList(self.draw_solid_faces)
	self.PLane_state_x=False
	self.PLane_state_y=False
	self.PLane_state_z=False
	self.layout = QVBoxLayout()
	self.setLayout(self.layout)
	self.Highlight_pressed=False
	self.highlighted_patch= []
	self.patch_size=0.8;
	self.legendCalls = []
	self.legendCalls.append(self.drawPatchColorBar)
	glEnable(GL_BLEND) 
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) 
	self.input_level= 0
	self.BlowuP = 0
	self.magnify_mode=False
	self.explosion=False
	self.Highlight_from_other_Modules=[]
	self.transFact=8;
	self.PatchBarList = DisplayList(self.drawPatchColorBar)
	self.axisLength = 100
	self.previousHighlightID=[]
	#self.CudaInitialization()
	#self.CudaProcessing()
	self.axisList = DisplayList(self.drawAxis)
	self.cubeList = DisplayList(self.drawCubes)
	self.highlightList = DisplayList(self.highlightDrawing)
	self.PatchColorChangeSignal.connect(self.cubeList.update)

	Template("default_patch_color", (0.2, 0.2, 0.2, 0.3))
	Template("patch_cmap", self.parent.agent.requestScene("patches").color_map)		

	self.doLegend()
	self.Process_planes()

	#Data structure default_color: list data structure to have a initial color.  
	self.default_color = [0.234, 0.1, 0.9, 0.12]

	self.values = None

    def  CudaInitialization(self):
	#Converting the list into numpy array for faster acess and putting it into the GPU for processing... 
	start = cuda.Event()
	end = cuda.Event()
	self.number_of_blocks=self.Array_Size/1024
	#Copying the values to the GPU!!! 
	#a_gpu = cuda.mem_alloc(self.np_values.nbytes)
	#cuda.memcpy_htod(a_gpu, self.np_values)
		
#Calculating the (Value-max)/max-min computation and storing it in a numpy array. Pre-calculating the maximum and minimum values.

# Space for the Kernel computation..

	func_mod = SourceModule("""
	#include<stdio.h>
// Needed to avoid name mangling so that PyCUDA can
// find the kernel function:
extern "C" {
    __global__ void func(float *a,int N,float minval,int denom)
    {
        //int idx = threadIdx.x+threadIdx.y*blockDim.y+blockIdx.x*blockDim.x;
	int idx =threadIdx.x+threadIdx.y*blockDim.x+blockIdx.x*(blockDim.x*blockDim.y)+blockIdx.x*blockDim.z+threadIdx.z*(blockDim.x*blockDim.y);
        if (idx < N)
	{       
   a[idx] = (a[idx]-minval)/denom;

}    }
}
""", no_extern_c=1)
	self.x = np.asarray(self.values, np.float32)
	func = func_mod.get_function('func')
	N = self.Array_Size
	x_gpu = gpuarray.to_gpu(self.x)
	h_minval = np.int32(self.minval)
	h_denom=np.int32(self.denominator)
	start.record()
	# a function to the GPU to caluculate the computation in the GPU.
	func(x_gpu.gpudata, np.uint32(N),np.uint32(h_minval),np.uint32(h_denom),block=(32,32,1),grid=(self.number_of_blocks+1,1,1))
	end.record() 
	end.synchronize()
	secs = start.time_till(end)*1e-3

	#print "SourceModule time and first three results:"
	print "%fs, %s" % (secs, str(self.x[:3]))
	#print 'x:       ', self.x[self.Array_Size-1]
	#print 'Func(x): ', x_gpu.get()[self.Array_Size-1],'Actual: ',(self.values[self.Array_Size-1]-self.minval)/(h_denom)
	self.x_colors=x_gpu.get()
    def calculateFPS(self):

 	   #Increase frame count
    	self.Frame_rate=self.Frame_rate+1

    #Get the number of milliseconds since glutInit called 
    #(or first call to glutGet(GLUT ELAPSED TIME)).
	self.currentTime =int(round(time.time() * 1000))
	#self.currentTime = glutGet(GLUT_ELAPSED_TIME);
	#print self.Frame_rate
	#print self.currentTime
    	    #Calculate time passed
    	timeInterval = self.currentTime - self.previousTime;
	if timeInterval > 1000:
		self.fps = self.Frame_rate / (timeInterval / 1000.0)
		self.previousTime = self.currentTime;
		self.Frame_rate=0


    def doLegend(self, bar_width = 20, bar_height = 160, bar_x = 20,
	bar_y = 90):
	"""Draws the legend information over the main view. This includes
	   the colorbars and anything done in fupanctions that have been
	   appeneded to the legendCalls member of this class.
	"""
	with overlays2D(self.width(), self.height(), self.bg_color):
	    for func in self.legendCalls:
		func()

    def createContent(self):

	self.layout.addWidget(self.Slider_in_PatchModule3D)
	self.layout.addItem(QSpacerItem(5,5))	
	self.layout.addWidget(self.CheckBox)
 	self.layout.addItem(QSpacerItem(5,5))	


    @Slot(ColorMap, tuple)
    def updateScene(self, patch_cmap, patch_range):
	"""Handle AttributeScene information from agent."""
	self.patch_cmap = patch_cmap
	self.patch_range=patch_range
	#self.cubeList.update()
	self.PatchBarList.update()
	self.doAxis()
	self.update()
	self.repaint()

    def PlaneUpdate(self):
	self.Cuda_OpenGLInteroperability()
	if self.enable_cuda:
            self.process_image()
	if not self.highlight_ids == []: 
		self.highlightList.update()	
	else:
		self.cubeList.update()
	self.repaint()

    def any(self,id,list):
	#print "The List:",list
	#print type(list)
	for i in range(list.__len__()-1):
		if id == list[i]:
			print "True"
			return True
    	return False	

    def changePlane_x(self,state):
	if (state):
		self.PLane_state_x=True
		self.PlaneUpdate()
		self.repaint()
	else:
		self.PLane_state_x=False
		self.PlaneUpdate()
		self.repaint()

    def changePlane_y(self,state):
	if (state):
		self.PLane_state_y=True
		self.PlaneUpdate()
		self.repaint()
	else:
		self.PLane_state_y=False
		self.PlaneUpdate()
		self.repaint()

    def changePlane_z(self,state):
	if (state):
		self.PLane_state_z=True
		self.PlaneUpdate()
		self.repaint()
	else:
		self.PLane_state_z=False
		self.PlaneUpdate()
		self.repaint()

    def changeTitle(self, state):
	if (state):
		self.trans= False
		self.PLane_state=True
		self.parent.Plane_x.show()
		self.parent.Plane_y.show()
		self.parent.Plane_z.show()		
		self.parent.Transp.show()
		self.PlaneUpdate()
	else:
		self.PLane_state=False
	   	self.trans= True
		self.parent.Plane_x.hide()
		self.parent.Plane_y.hide()
		self.parent.Plane_z.hide()
		self.parent.Transp.hide()
		self.PlaneUpdate()
		#self.parent.Slider_for_Spread.hide()
	self.repaint()
    def drawAxis(self):
	"""This function does the actual drawing of the lines in the axis."""
	glLineWidth(2.0)
	with glSection(GL_LINES):
	    glColor4f(1.0, 0.0, 0.0, 1.0)
	    glVertex3f(0, 0, 0)
	    glVertex3f(self.axisLength, 0, 0)

	    glColor4f(0.0, 1.0, 0.0, 1.0)
	    glVertex3f(0, 0, 0)
	    glVertex3f(0, -self.axisLength, 0)

	    glColor4f(0.0, 0.0, 1.0, 1.0)
	    glVertex3f(0, 0, 0)
	    glVertex3f(0, 0, -self.axisLength)

#Draws the patches when there is no patch selected by the user. 
    #@profile

    def process(self,width, height):
	    """ Use PyCuda """
	    grid_dimensions   = (32,32)

	    self.source_mapping = self.pycuda_source_pbo.map()
	    self.dest_mapping   = self.pycuda_dest_pbo.map()

	    self.invert.prepared_call(grid_dimensions, (16, 16, 1),
		    self.source_mapping.device_ptr(),
		    self.dest_mapping.device_ptr())

	    self.cuda_driver.Context.synchronize()

	    self.source_mapping.unmap()
	    self.dest_mapping.unmap()

    def process_image(self):
	    """ copy image and process using CUDA """
	    image_width, image_height = 512,512
	    assert self.source_pbo is not None

	    # tell cuda we are going to get into these buffers
	    self.pycuda_source_pbo.unregister()

	    # activate destination buffer
	    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(self.source_pbo))

	    # read data into pbo. note: use BGRA format for optimal performance
	    import OpenGL.raw.GL as rawgl

	    rawgl.glReadPixels(
		     0,                  #start x
		     0,                  #start y
		     image_width,        #end   x
		     image_height,       #end   y
		     GL_BGRA,            #format
		     GL_UNSIGNED_BYTE,   #output type
		     ctypes.c_void_p(0))

	    self.pycuda_source_pbo = self.cuda_gl.BufferObject(long(self.source_pbo))

	    # run the Cuda kernel
	    self.process(image_width, image_height)
	    # blit convolved texture onto the screen
	    # download texture from PBO
	    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, long(self.dest_pbo))
	    glBindTexture(GL_TEXTURE_2D, self.output_texture)

	    rawgl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
		            image_width, image_height,
		            GL_BGRA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
    def drawCubes(self):

	if self.flag:
		self.Range()
		self.flag=False
		self.X_interval=self.cx_min_value+self.interval_div_x
		self.Y_interval=self.cy_min_value+self.interval_div_y
		self.Z_interval=self.cz_min_value+self.interval_div_z
	if self.values is None:
	    return
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)# Clear Screen And Depth Buffe    # Reset The Modelview Matrix
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
  	i=0
	mdl = np.array(glGetDoublev(GL_MODELVIEW_MATRIX)).flat
        camera = np.array([-(mdl[0] * mdl[12] + mdl[1] * mdl[13] + mdl[2] * mdl[14]),
                            -(mdl[4] * mdl[12] + mdl[5] * mdl[13] + mdl[6] * mdl[14]),
                            -(mdl[8] * mdl[12] + mdl[9] * mdl[13] + mdl[10] * mdl[14])])
	print "mdl",mdl,"\n"
	print np.linalg.norm(camera - 1)
	glPushMatrix()
	for patch, value in zip(self.patchid, self.values):  
		cx, cy, cz, sx, sy, sz, level, owner, avghops, maxhops, minhops = self.parent.agent.patch_coords_dict[patch]
		if level != self.input_level: 
					continue
		if self.PlaneValues[0] == True:		
			if cx < (self.X_interval):
					continue
		if self.PlaneValues[1] == True:
				if cx < (self.X_interval*2):
					continue
		if self.PlaneValues[2] == True:
				if cx > (self.X_interval*3):
					continue
		if self.PlaneValues[3] == True:
				if cx > (self.X_interval*4):
					continue
		if self.PlaneValues[4] == True:		
			if cy < (self.Y_interval):
					continue
		if self.PlaneValues[5] == True:
				if cy < (self.Y_interval*2):
					continue
		if self.PlaneValues[6] == True:
				if cy > (self.Y_interval*3):
					continue
		if self.PlaneValues[7] == True:
				if cy > (self.Y_interval*4):
					continue
		if self.PlaneValues[8] == True:		
			if cz < (self.Z_interval):
					continue
		if self.PlaneValues[9] == True:
				if cz < (self.Z_interval*2):
					continue
		if self.PlaneValues[10] == True:
				if cz > (self.Z_interval*3):
					continue
		if self.PlaneValues[11] == True:
				if cz > (self.Z_interval*4):
					continue
		glPushMatrix()
		self.patch_colors=self.patch_cs[i]
		i=i+1
		glTranslatef(-cx, -cy, -cz)
		glScalef(sx, sy, sz)
		#print self.transFact
		glColor4f(self.patch_colors[0],self.patch_colors[1],self.patch_colors[2],self.transFact*0.1)
		if level == self.input_level:
			if self.Wire_Frame:
				glutSolidCube(self.patch_size)
			else:
	    			glutWireCube(self.patch_size)
	
		glPopMatrix()  
	if self.PLane_state:
			if self.PLane_state_x:
				glPushMatrix()
				glColor4f(0.7, 0.0, 0.0,0.3)
				glTranslate(-(self.cx_min_value+self.interval_div_x),0,0)
				if self.PlaneValues[0] == False:
					glBegin(GL_QUADS)
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,2.5);
					glVertex3f(0,2.5,2.5);
					glVertex3f(0,2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()
				glTranslate(-(self.cx_min_value+(self.interval_div_x)),0,0)
				if self.PlaneValues[1] == False:
					glBegin(GL_QUADS)
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,2.5);
					glVertex3f(0,2.5,2.5);
					glVertex3f(0,2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()
				glTranslate(-(self.cx_min_value+(self.interval_div_x)),0,0)
				if self.PlaneValues[2] == False:				
					glBegin(GL_QUADS)
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,2.5);
					glVertex3f(0,2.5,2.5);
					glVertex3f(0,2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glTranslate(-(self.cx_min_value+(self.interval_div_x)),0,0)
				if self.PlaneValues[3] == False:
					glBegin(GL_QUADS)
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,2.5);
					glVertex3f(0,2.5,2.5);
					glVertex3f(0,2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glPopMatrix()
			if self.PLane_state_y:
				glPushMatrix()
				glColor4f(0.7, 0.0, 0.0,0.3)
				glTranslate(0,-(self.cy_min_value+self.interval_div_y),0)
				if self.PlaneValues[4] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,2.5);
					glVertex3f(2.5,0,2.5);
					glVertex3f(2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glTranslate(0,-(self.cy_min_value+self.interval_div_y),0)
				if self.PlaneValues[4] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,2.5);
					glVertex3f(2.5,0,2.5);
					glVertex3f(2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glTranslate(0,-(self.cy_min_value+self.interval_div_y),0)
				if self.PlaneValues[5] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,2.5);
					glVertex3f(2.5,0,2.5);
					glVertex3f(2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glTranslate(0,-(self.cy_min_value+self.interval_div_y),0)
				if self.PlaneValues[6] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,2.5);
					glVertex3f(2.5,0,2.5);
					glVertex3f(2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()		
				glPopMatrix()
			if self.PLane_state_z:	
				glPushMatrix()
				glColor4f(0.7, 0.0, 0.0,0.3)
				glTranslate(0,0,-(self.cz_min_value+self.interval_div_z))
				if self.PlaneValues[7] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,2.5,0);
					glVertex3f(2.5,2.5,0);
					glVertex3f(2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glEnd()	
				glTranslate(0,0,-(self.cz_min_value+self.interval_div_z))
				if self.PlaneValues[8] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,2.5,0);
					glVertex3f(2.5,2.5,0);
					glVertex3f(2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glEnd()	
				glTranslate(0,0,-(self.cz_min_value+self.interval_div_z))
				if self.PlaneValues[9] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,2.5,0);
					glVertex3f(2.5,2.5,0);
					glVertex3f(2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glEnd()	
				glTranslate(0,0,-(self.cz_min_value+self.interval_div_z))
				if self.PlaneValues[10] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,2.5,0);
					glVertex3f(2.5,2.5,0);
					glVertex3f(2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glEnd()		
				glPopMatrix()
	self.do_tick()		
	glPopMatrix()

#Finds the z values of the rendered patches to implement the z-order buffering.
    def Find_z_Value(self):
	Z_Values=[]
	c={}

	for ids,value in zip(self.patchid, self.values):
		cx, cy, cz, sx, sy, sz, level, owner, avghops, maxhops, minhops= self.parent.agent.patch_coords_dict[ids]	
  		c.update({cz:ids})

	Z_Values= SortedDisplayDict(c)
	#print Z_Values

	return Z_Values


#The function that is put in a display list. 
#This is responsible for visualizing the Repatches and their neighbours as well as the magnify effect. They also govern the visualization of the #slice and dice operation.  
    #@profile
    def highlightDrawing(self):
	if self.values is None:
	    return
	keys = []	
	self.Find_z_Value()
	c=0
	i=0
	mdl = np.array(glGetDoublev(GL_MODELVIEW_MATRIX)).flat
        camera = np.array([-(mdl[0] * mdl[12] + mdl[1] * mdl[13] + mdl[2] * mdl[14]),
                            -(mdl[4] * mdl[12] + mdl[5] * mdl[13] + mdl[6] * mdl[14]),
                            -(mdl[8] * mdl[12] + mdl[9] * mdl[13] + mdl[10] * mdl[14])])
	print "mdl",mdl,"\n",camera
	print "sorting!!!!!!!!!!"

	glPushMatrix()	
		#a = self.patch_cmap.getColor((value-minval) / denominator) 	
		#self.patch_colors=list(a)
	for ids,value in zip(self.patchid, self.values): 
			cx, cy, cz, sx, sy, sz, level, owner, avghops, maxhops, minhops= self.parent.agent.patch_coords_dict[ids]
			if level != self.input_level: 
						continue			
			if self.PlaneValues[0] == True:		
				if cx < (self.X_interval):
						continue
			if self.PlaneValues[1] == True:	
					if cx < (self.X_interval*2):
						continue
			if self.PlaneValues[2] == True:
					if cx > (self.X_interval*3):
						continue
			if self.PlaneValues[3] == True:
					if cx > (self.X_interval*4):
						continue
			if self.PlaneValues[4] == True:		
				if cy < (self.Y_interval):
						continue
			if self.PlaneValues[5] == True:
					if cy < (self.Y_interval*2):
						continue
			if self.PlaneValues[6] == True:
					if cy > (self.Y_interval*3):
						continue
			if self.PlaneValues[7] == True:
					if cy > (self.Y_interval*4):
						continue
			if self.PlaneValues[8] == True:		
				if cz < (self.Z_interval):
						continue	
			if self.PlaneValues[9] == True:
					if cz < (self.Z_interval*2):
						continue
			if self.PlaneValues[10] == True:
					if cz > (self.Z_interval*3):
						continue
			if self.PlaneValues[11] == True:
					if cz > (self.Z_interval*4):
						continue

			glPushMatrix()			
			self.patch_colors=self.patch_cs[i]	
			i=i+1		
			#self.patch_colors= self.patch_cmap.getColor((value-minval) / denominator)
			glTranslatef(-cx, -cy, -cz)
			glScalef(sx, sy, sz)
			glColor4f(self.patch_colors[0],self.patch_colors[1],self.patch_colors[2],self.transFact*0.1) 	
			if self.highlight_ids[-1] == []:
				glColor4f(self.patch_colors[0],self.patch_colors[1],self.patch_colors[2],self.transFact*0.1) 

			if not self.magnify_mode or (self.magnify_mode and not self.neighbours) and not (self.highlight_ids[-1] < self.Max_No_of_Patches):
				if self.Wire_Frame:
						glutSolidCube(self.patch_size)
				else:
	    				glutWireCube(self.patch_size)
				if not self.propagate:
					if ids in self.highlight_ids:
						glColor4f(self.patch_colors[0],self.patch_colors[1],self.patch_colors[2],1)
						if(self.patch_size < 0.5):					
							glutSolidCube(0.55)
						else:
							glutSolidCube(self.patch_size+(self.patch_size*0.1))

			if self.magnify_mode:
				#I am using a new mode called magnify where the developers could magnify the patch of interest and the other patches are relatively smaller than the patch of interest.The patch of interest is blown up to an extent of factor, The neighbours are suppressed to a size relative as a function of their distance from the centre of the patch of interest.				
				if ids in self.highlight_ids:
					glColor4f(self.patch_colors[0],self.patch_colors[1],self.patch_colors[2],1)
					if ids in self.highlight_ids[-1]:
						glutSolidCube(1.3)
					else:	
						if self.highlighted_patch == []:
							#self.highlighted_patch.append({ids:[cx,cx,cy]})
							Difference_x=abs(self.patch_x-cx)
							Difference_y=abs(self.patch_y-cy)
							Difference_z=abs(self.patch_z-cz)		
							XAxis_coord=(self.patch_x-cx)/Difference_x
							YAxis_coord=(self.patch_y-cy)/Difference_y
							ZAxis_coord=(self.patch_z-cz)/Difference_z
							glTranslatef((0.25)*XAxis_coord ,(0.25)*YAxis_coord,(0.25)*ZAxis_coord)
							glutSolidCube(0.65)
				elif self.neighbours:
					if ids in self.neighbours:
						Difference_x=abs(self.patch_x-cx)
						Difference_y=abs(self.patch_y-cy)
						Difference_z=abs(self.patch_z-cz)
						Distance=round(Difference_x+Difference_y+Difference_z)
						XAxis_coord=(self.patch_x-cx)/Difference_x
						YAxis_coord=(self.patch_y-cy)/Difference_y
						ZAxis_coord=(self.patch_z-cz)/Difference_z
						glTranslatef((0.25+ self.transFact)*XAxis_coord ,(0.25+ self.transFact)*YAxis_coord,(0.25+self.transFact)*ZAxis_coord)
						glColor4f(self.patch_colors[1],0,0,(1/Distance)*2.5)
						if self.Wire_Frame:
								glutSolidCube((1/Distance)*(2.5+1+(self.input_level*0.8)))
						else:
								glutWireCube((1/Distance)*(2.5+1+(self.input_level*0.8)))
			else :
					if self.propagate: 
						if ids in self.new:	
							c=c+1
							glColor4f(self.patch_colors[1],0,0,1)
							if(self.patch_size < 0.5):					
								glutSolidCube(0.55 + ((self.patch_size)*0.1))
							else:
								glutSolidCube(self.patch_size+(self.patch_size*0.1))
					else:
						if self.neighbours:
							if ids in self.neighbours:	
								c=c+1
								glColor4f(self.patch_colors[1],0,0,1)
								if(self.patch_size < 0.5):					
									glutSolidCube(0.55 + ((self.patch_size)*0.1))
								else:
									glutSolidCube(self.patch_size+(self.patch_size*0.1))
			glPopMatrix()

	if self.PLane_state:
			if self.PLane_state_x:
				glPushMatrix()
				glColor4f(0.7, 0.0, 0.0,0.3)
				glTranslate(-(self.cx_min_value+self.interval_div_x),0,0)
				if self.PlaneValues[0] == False:
					glBegin(GL_QUADS)
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,2.5);
					glVertex3f(0,2.5,2.5);
					glVertex3f(0,2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()
				glTranslate(-(self.cx_min_value+(self.interval_div_x)),0,0)
				if self.PlaneValues[1] == False:
					glBegin(GL_QUADS)
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,2.5);
					glVertex3f(0,2.5,2.5);
					glVertex3f(0,2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()
				glTranslate(-(self.cx_min_value+(self.interval_div_x)),0,0)
				if self.PlaneValues[2] == False:				
					glBegin(GL_QUADS)
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,2.5);
					glVertex3f(0,2.5,2.5);
					glVertex3f(0,2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glTranslate(-(self.cx_min_value+(self.interval_div_x)),0,0)
				if self.PlaneValues[3] == False:
					glBegin(GL_QUADS)
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,2.5);
					glVertex3f(0,2.5,2.5);
					glVertex3f(0,2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glPopMatrix()
			if self.PLane_state_y:
				glPushMatrix()
				glColor4f(0.7, 0.0, 0.0,0.3)
				glTranslate(0,-(self.cy_min_value+self.interval_div_y),0)
				if self.PlaneValues[4] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,2.5);
					glVertex3f(2.5,0,2.5);
					glVertex3f(2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glTranslate(0,-(self.cy_min_value+self.interval_div_y),0)
				if self.PlaneValues[4] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,2.5);
					glVertex3f(2.5,0,2.5);
					glVertex3f(2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glTranslate(0,-(self.cy_min_value+self.interval_div_y),0)
				if self.PlaneValues[5] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,2.5);
					glVertex3f(2.5,0,2.5);
					glVertex3f(2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glTranslate(0,-(self.cy_min_value+self.interval_div_y),0)
				if self.PlaneValues[6] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,2.5);
					glVertex3f(2.5,0,2.5);
					glVertex3f(2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()		
				glPopMatrix()
			if self.PLane_state_z:	
				glPushMatrix()
				glColor4f(0.7, 0.0, 0.0,0.3)
				glTranslate(0,0,-(self.cz_min_value+self.interval_div_z))
				if self.PlaneValues[7] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,2.5,0);
					glVertex3f(2.5,2.5,0);
					glVertex3f(2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glEnd()	
				glTranslate(0,0,-(self.cz_min_value+self.interval_div_z))
				if self.PlaneValues[8] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,2.5,0);
					glVertex3f(2.5,2.5,0);
					glVertex3f(2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glEnd()	
				glTranslate(0,0,-(self.cz_min_value+self.interval_div_z))
				if self.PlaneValues[9] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,2.5,0);
					glVertex3f(2.5,2.5,0);
					glVertex3f(2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glEnd()	
				glTranslate(0,0,-(self.cz_min_value+self.interval_div_z))
				if self.PlaneValues[10] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,2.5,0);
					glVertex3f(2.5,2.5,0);
					glVertex3f(2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glEnd()		
				glPopMatrix()	

	self.do_tick()
	glPopMatrix()

    def do_tick(self):
	    if ((time.clock () * 1000.0) - self.time_of_last_titleupdate >= 1000.):
		self.frames_per_second = self.frame_counter                   # Save The FPS
		self.frame_counter = 0  # Reset The FPS Counter
		szTitle = "%d FPS" % (self.frames_per_second )
		self.setWindowTitle( szTitle )
		print "asda",szTitle
		self.time_of_last_titleupdate = time.clock () * 1000.0
	    self.frame_counter += 1

    def mousePressEvent(self, event):
	"""We capture right clicking for picking here."""

	super(Patch3dGLWidget, self).mousePressEvent(event)
	if event.button() == Qt.RightButton:
		print "\nDoPick\n:"
		#profile.runctx('self.doPick (event)', globals(), locals())
		ID=self.doPick(event)
		if ID > self.Max_No_of_Patches:
			#Converting a List into an integer Value 
			PLane_No=ID-self.Max_No_of_Patches
			self.PLane_highlight(PLane_No)	
			self.highlightList.update()		
			return 
		self.var = [["patches", self.doPick(event)]]
		#Propagating the highlights to the other modules in place.
		self.parent.agent.selectionChanged(self.var)

		if not self.highlight_ids == []:
			self.highlightList.update()
		else:
			self.cubeList.update()
		self.repaint()
		glGetError()
	if event.button() == Qt.LeftButton:	
		return

    def map_node_color(self, val, preempt_range = 0):
	"""Turns a color value in [0,1] into a 4-tuple RGBA listcolor.
	   Used to map nodes.
	"""
	return self.patch_cmap.getColor(val, preempt_range)


#Accumalates all the integers in a given list
#Finds the the first number in the list.
    def magic(self,number):
    	return int(''.join(str(i) for i in number))


#highlights the selected using a hashmap in the plane

    def face_rendering(self):

    	print "Face"


    def	PLane_highlight(self,Plane_no):

	self.PlaneValues[Plane_no-1]=True

	#for i in range(12):
		#print " \n",self.PlaneValues[i]

#Code from Torus module of Boxfish.Allows to select a patch in the application domain. 
    def changeValue_for_propagate(self):
	print self.propagate
    
    def doPick(self, event):
	"""Allow the user to pick nodes."""
	# Adapted from Josh Levine's version in Boxfish 0.1
	#steps:
	#render the scene with labeled patches
	#find the color of the pixel @self.x, self.y
	#map color back to id and return
	self.currentTime = glutGet(GLUT_ELAPSED_TIME);
	self.previousTime=self.currentTime
	#disable unneded
	glDisable(GL_LIGHTING)
	glDisable(GL_LIGHT0)
	glDisable(GL_BLEND)

	#set up the selection buffer
	#Taking the Select_buf_size to be the no of patches(approximating the values)
	select_buf_size = self.MaxID + 12
	self.Max_No_of_Patches=self.MaxID

	glSelectBuffer(select_buf_size)

	#switch to select mode
	glRenderMode(GL_SELECT)

	#initialize name stack
	glInitNames()
	glPushName(0)

	#set up the pick matrix to draw a narrow view
	viewport = glGetIntegerv(GL_VIEWPORT)
	glMatrixMode(GL_PROJECTION)
	glPushMatrix()
	glLoadIdentity()
	#this sets the size and location of the pick window
	#changing the 1,1 will change sensitivity of the pick
	gluPickMatrix(event.x(),(viewport[3]-event.y()),
	    1,1,viewport)
	set_perspective(self.fov, self.width()/float(self.height()),
	    self.near_plane, self.far_plane)
	#switch back to modelview and draw the scene
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	glTranslatef(*self.translation[:3])
	glMultMatrixd(self.rotation)
	#let's draw some red boxes, color is inconsequential
	glColor3f(1.0, 0.0, 0.0)
	# Redo the drawing
      	glPushMatrix()			

	for patch,values in zip(self.patchid, self.values):
	    	#The different attributes are the inputts onto the variables Using the Dictionary.
		cx, cy, cz, sx, sy, sz, level, owner, avghops, maxhops, minhops = self.parent.agent.patch_coords_dict[patch]
		if self.PlaneValues[0] == True:		
			if cx < (self.X_interval):
					continue
		if self.PlaneValues[1] == True:
				if cx < (self.X_interval*2):
					continue
		if self.PlaneValues[2] == True:
				if cx > (self.X_interval*3):
					continue
		if self.PlaneValues[3] == True:
				if cx > (self.X_interval*4):
					continue
		if self.PlaneValues[4] == True:		
			if cy < (self.Y_interval):
					continue
		if self.PlaneValues[5] == True:
				if cy < (self.Y_interval*2):
					continue
		if self.PlaneValues[6] == True:
				if cy > (self.Y_interval*3):
					continue
		if self.PlaneValues[7] == True:
				if cy > (self.Y_interval*4):
					continue
		if self.PlaneValues[8] == True:		
			if cz < (self.Z_interval):
					continue
		if self.PlaneValues[9] == True:
				if cz < (self.Z_interval*2):
					continue
		if self.PlaneValues[10] == True:
				if cz > (self.Z_interval*3):
					continue
		if self.PlaneValues[11] == True:
				if cz > (self.Z_interval*4):
					continue

		glLoadName(patch)
		glPushMatrix()
		glTranslatef(-cx, -cy, -cz)
	     	glScalef(sx, sy, sz)
		if not (self.trans):
			if level == self.input_level:
				glTranslatef(-self.BlowuP,-self.BlowuP,0)
		if level == self.input_level: 
				if self.Wire_Frame:
						glutSolidCube(self.patch_size)
				else:
	    				glutWireCube(self.patch_size)
	
	    	glPopMatrix() 

	if self.PLane_state:
			if self.PLane_state_x:
				glPushMatrix()
				glColor4f(0.7, 0.0, 0.0,0.3)
				#To make Sure that the patche IDS do no coincide with the Planes.To retrieve the plane IDS we would have Find the remainder of the hitlist. 
				glTranslate(-(self.cx_min_value+self.interval_div_x),0,0)
				glLoadName(self.Max_No_of_Patches+1)
				if self.PlaneValues[0] == False:	
					glBegin(GL_QUADS)
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,2.5);
					glVertex3f(0,2.5,2.5);
					glVertex3f(0,2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()
				glTranslate(-(self.cx_min_value+(self.interval_div_x)),0,0)
				glLoadName(self.Max_No_of_Patches+2);
				if self.PlaneValues[1] == False:
					glBegin(GL_QUADS)
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,2.5);
					glVertex3f(0,2.5,2.5);
					glVertex3f(0,2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()
				glTranslate(-(self.cx_min_value+(self.interval_div_x)),0,0)
				glLoadName(self.Max_No_of_Patches+3);
				if self.PlaneValues[2] == False:
					glBegin(GL_QUADS)
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,2.5);
					glVertex3f(0,2.5,2.5);
					glVertex3f(0,2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glTranslate(-(self.cx_min_value+(self.interval_div_x)),0,0)
				glLoadName(self.Max_No_of_Patches+4);
				if self.PlaneValues[3] == False:
					glBegin(GL_QUADS)
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,2.5);
					glVertex3f(0,2.5,2.5);
					glVertex3f(0,2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(0,-(self.cy_max_value-self.cy_min_value)-2.5,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glPopMatrix()
			if self.PLane_state_y:
				glPushMatrix()
				glColor4f(0.7, 0.0, 0.0,0.3)
				glTranslate(0,-(self.cy_min_value+self.interval_div_y),0)
				glLoadName(self.Max_No_of_Patches+5);
				if self.PlaneValues[4] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,2.5);
					glVertex3f(2.5,0,2.5);
					glVertex3f(2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glTranslate(0,-(self.cy_min_value+self.interval_div_y),0)
				glLoadName(self.Max_No_of_Patches+6);				
				if self.PlaneValues[5] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,2.5);
					glVertex3f(2.5,0,2.5);
					glVertex3f(2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glTranslate(0,-(self.cy_min_value+self.interval_div_y),0)
				glLoadName(self.Max_No_of_Patches+7);				
				if self.PlaneValues[6] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,2.5);
					glVertex3f(2.5,0,2.5);
					glVertex3f(2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()	
				glTranslate(0,-(self.cy_min_value+self.interval_div_y),0)
				glLoadName(self.Max_No_of_Patches+8);				
				if self.PlaneValues[7] == False:
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,2.5);
					glVertex3f(2.5,0,2.5);
					glVertex3f(2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,0,-(self.cz_max_value-self.cz_min_value)-2.5);
					glEnd()		
				glPopMatrix()
			if self.PLane_state_z:	
				glPushMatrix()
				glColor4f(0.7, 0.0, 0.0,0.3)
				glTranslate(0,0,-(self.cz_min_value+self.interval_div_z))
				glLoadName(self.Max_No_of_Patches+9);				
				if self.PlaneValues[8] == False:					
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,2.5,0);
					glVertex3f(2.5,2.5,0);
					glVertex3f(2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glEnd()	
				glTranslate(0,0,-(self.cz_min_value+self.interval_div_z))
				glLoadName(self.Max_No_of_Patches+10);				
				if self.PlaneValues[9] == False:					
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,2.5,0);
					glVertex3f(2.5,2.5,0);
					glVertex3f(2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glEnd()	
				glTranslate(0,0,-(self.cz_min_value+self.interval_div_z))
				glLoadName(self.Max_No_of_Patches+11);				
				if self.PlaneValues[10] == False:					
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,2.5,0);
					glVertex3f(2.5,2.5,0);
					glVertex3f(2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glEnd()	
				glTranslate(0,0,-(self.cz_min_value+self.interval_div_z))
				glLoadName(self.Max_No_of_Patches+12);				
				if self.PlaneValues[11] == False:					
					glBegin(GL_QUADS)
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,2.5,0);
					glVertex3f(2.5,2.5,0);
					glVertex3f(2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glVertex3f(-(self.cx_max_value-self.cx_min_value)-2.5,-(self.cy_max_value-self.cy_min_value)-2.5,0);
					glEnd()		
				glPopMatrix()

	glMatrixMode(GL_PROJECTION)
	glPopMatrix()
	glFlush()
	#get the hit buffer
	glMatrixMode(GL_MODELVIEW)
	pick_buffer = glRenderMode(GL_RENDER)
	nearest = 4294967295
	hitlist = []
	for hit in pick_buffer :
	    if hit[0] < nearest :
	      nearest = hit[0]
	      hitlist = [hit[2][0]]	

	#go back to normal rendering
	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
	glEnable(GL_BLEND)
   	glPopMatrix()
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	if hitlist > self.Max_No_of_Patches:
		#Converting a List into an integer Value 
		hitlist1=self.magic(hitlist)-self.Max_No_of_Patches
		#print "Patch ID Selected: ",hitlist1

	#else:
		#print "Patch ID Selected: ",hitlist

	return hitlist

    def initPatch(self,cx,cy,cz,sx,sy,sz,level, owner, avghops, maxhops, minhops):
	#self.msgBox = QMessageBox()
	var = str("\nPatch-Information\nCenterx : "+str(cx)+"\nCentery : "+str(cy)+"\nCenterz : "+str(cz)+"\nSizex : "+str(sx)+"\nSizey : "+str(sy)+"\nSizez : "+str(sz)+"\nLevel :"+str(level))
	var2 = str("\nOwner(MPI rank):"+str(owner)+"\nAverage-Hops :"+str(avghops)+"\nMaximum-Hops :"+str(maxhops)+"\nMinimum-Hops :"+str(minhops))

	#self.msgBox.setText(var + var2)
	#self.msgBox.exec_()     

#Finds the maximum and minimum x,y and z values bounded by the simulated domain.
#Calculates the area where the slice planes need to be placed in the simulated domain. 


    def Range(self):
		c=0
		self.patch_cs=[]

		self.Array_Size=len(self.patchid)
		self.cx_p=self.cy_p=self.cz_p=np.zeros(self.Array_Size)
		
		self.Dict_Values=np.zeros((self.Array_Size,12))
		self.minval = int(round(min(self.values)))
		self.maxval = int(round(max(self.values)))
		self.denominator = self.maxval-self.minval
		if self.denominator == 0:
			self.denominator = 1
		#Calculating the number of blocks and threads in the program!! 
		#For example if the data is 3871 there can be 3 blocks of 1024 and 1 block of 759!!
		#They can be alloted by <<Compute>>3,1024<< and compute<<1,759<<.
		#print "==========  Number of Blocks=>",number_of_blocks,"\n========= number_of_threads=>:",number_of_threads
		#self.patch_attributes=[] 
		self.CudaInitialization()
		MaxID=0
		#names = ['patch-center-x','patch-center-y','patch-center-z','patch-size-x','patch-size-y','patch-size-z','level','Avg Hops-Extents','Max Hops-Extents','Min Hops-Extents','Owner-Extents']
		#formats = ['f4','f4','f4','f4','f4','f4','i8','f8','f8','f8','f8']
		#dtype = dict(names = names, formats=formats)
		#self.Dict_Values = np.fromiter(self.parent.agent.patch_coords_dict.itervalues(), dtype=dtype,count=len(self.parent.agent.patch_coords_dict))	
		self.patchid = np.array((self.patchid))
		cx_min_value=-1
		cx_max_value=20000000
		cy_min_value=-1
		cy_max_value=20000000
		cz_min_value=-1	
		cz_max_value=20000000
		Value_index=-1
		glPushMatrix()		
		for patch, value in zip(self.patchid, self.values):  	
			cx, cy, cz, sx, sy, sz, level, owner, avghops, maxhops, minhops = self.parent.agent.patch_coords_dict[patch]
			Value_index=Value_index+1
			#self.Dict_Values[Value_index][0],self.Dict_Values[Value_index][1],self.Dict_Values[Value_index][2],self.Dict_Values[Value_index][3],self.Dict_Values[Value_index][4],self.Dict_Values[Value_index][5],self.Dict_Values[Value_index][6],self.Dict_Values[Value_index][7],self.Dict_Values[Value_index][8],self.Dict_Values[Value_index][9],self.Dict_Values[Value_index][10],self.Dict_Values[Value_index][11]=patch,cx, cy, cz, sx, sy, sz, level, owner, avghops, maxhops, minhops
			#self.cx_p[Value_index]=cx
			#self.cy_p[Value_index]=cy
			#self.cz_p[Value_index]=cz
			if level != self.input_level: 
					continue
			if self.PlaneValues[0] == True:		
				if cx < (self.X_interval):
						continue
			if self.PlaneValues[1] == True:
					if cx < (self.X_interval*2):
						continue
			if self.PlaneValues[2] == True:
					if cx > (self.X_interval*3):
						continue
			if self.PlaneValues[3] == True:
					if cx > (self.X_interval*4):
						continue
			if self.PlaneValues[4] == True:		
				if cy < (self.Y_interval):
						continue
			if self.PlaneValues[5] == True:
					if cy < (self.Y_interval*2):
						continue
			if self.PlaneValues[6] == True:
					if cy > (self.Y_interval*3):
						continue
			if self.PlaneValues[7] == True:
					if cy > (self.Y_interval*4):
						continue
			if self.PlaneValues[8] == True:		
				if cz < (self.Z_interval):
						continue
			if self.PlaneValues[9] == True:
					if cz < (self.Z_interval*2):
						continue
			if self.PlaneValues[10] == True:
					if cz > (self.Z_interval*3):
						continue
			if self.PlaneValues[11] == True:
					if cz > (self.Z_interval*4):
						continue
			#self.patch_cs.append(self.patch_cmap.getColor((value-self.minval) / self.denominator))
			self.patch_cs.append(self.patch_cmap.getColor(self.x_colors[Value_index]))
			c=c+1
			if MaxID < patch:
				MaxID=patch	
			if cx_min_value < cx:
				cx_min_value=cx
			if cx_max_value > cx:
				cx_max_value=cx
			if cy_min_value < cy:
				cy_min_value=cy
			if cy_max_value > cy:
				cy_max_value=cy
			if cz_min_value < cz:
				cz_min_value=cz
			if cz_max_value > cz:
				cz_max_value=cz
		glPopMatrix()		
		#print self.Dict_Values[	self.Array_Size-1][11]
		print "I am in Range"
		self.interval_div_x=int(round(abs(cx_min_value-cx_max_value)/5))		
		self.interval_div_y=int(round(abs(cy_min_value-cy_max_value)/5))
		self.interval_div_z=int(round(abs(cz_min_value-cz_max_value)/5))
		self.cx_min_value=cx_max_value
		self.cx_max_value=cx_min_value
		self.cy_min_value=cy_max_value
		self.cy_max_value=cy_min_value
		self.cz_min_value=cz_max_value
		self.cz_max_value=cz_min_value
		self.MaxID=MaxID
		return MaxID

#Creates a class variable to inform the Widget about the mode in which it is operating.

    def Magnifymode(self,Switch_state):
	#print Switch_state
	self.transFact=0
	self.parent.Transp.setValue(self.transFact*.1)    
	if Switch_state:
		self.magnify_mode=True	
	else: 	
		self.magnify_mode=False


#Given a highlighted patch id,it reinitturns the patchids that are to be about to magnified.


    def Cuda_OpenGLInteroperability(self):
	    #setup pycuda gl interop
	    import pycuda.gl.autoinit
    	    import pycuda.gl
	    self.cuda_gl = pycuda.gl
	    self.cuda_driver = pycuda.driver

	    cuda_module = SourceModule("""
	    __global__ void invert(unsigned char *source, unsigned char *dest)
	    {
	      int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
	      int thread_num       = threadIdx.y * blockDim.x + threadIdx.x;
	      int threads_in_block = blockDim.x * blockDim.y;
	      //Since the image is RGBA we multiply the index 4.
	      //We'll only use the first 3 (RGB) channels though
	      int idx              = 4 * (threads_in_block * block_num + thread_num);
	      dest[idx  ] = 255 - source[idx  ];
	      dest[idx+1] = 255 - source[idx+1];
	      dest[idx+2] = 255 - source[idx+2];
	      dest[idx+3] = 0.04- source[idx+3];
	    }
	    """)
	    self.invert = cuda_module.get_function("invert")
	    # The argument "PP" indicates that the invert function will take two PBOs as arguments
	    self.invert.prepare("PP")
	    self.create_texture(*self.initial_size)
	    # create source and destination pixel buffer objects for processing
	    self.create_PBOs(*self.initial_size)

    def Cuda_Distance(self):	
	start = cuda.Event()
	end = cuda.Event()
	func_mod = SourceModule("""
	#include<stdio.h>
	#include<stdlib.h>
	#include<math.h>
// Needed to avoid name mangling so that PyCUDA can
// find the kernel function:
extern "C" {
    __global__ void func(float *a,float *b,float *c,int N,float x,float y,float z)
    {
        int idx = threadIdx.x+threadIdx.y*blockDim.y+blockIdx.x*blockDim.x;
        if (idx < N)
	{       
   	a[idx] = roundf(abs(a[idx]-x)+abs(b[idx]-y)+abs(c[idx]-z));
}    }
}
""", no_extern_c=1)
	func = func_mod.get_function('func')
	N = self.Array_Size
	x_gpu = gpuarray.to_gpu(np.asarray(self.cx_p, np.float32))
	y_gpu = gpuarray.to_gpu(np.asarray(self.cy_p, np.float32))
	z_gpu = gpuarray.to_gpu(np.asarray(self.cz_p, np.float32))
	x = np.float32(self.patch_x)
	y = np.float32(self.patch_y)
	z = np.float32(self.patch_z)
	h_denom=np.int32(self.denominator)
	start.record()
	# a function to the GPU to caluculate the computation in the GPU.
	func(x_gpu.gpudata,y_gpu.gpudata,z_gpu.gpudata,np.uint32(N),x,y,z,block=(1024,1, 1),grid=(self.number_of_blocks+1,1,1))
	end.record() 
	end.synchronize()
	secs = start.time_till(end)*1e-3
	print x_gpu.get()
	print "SourceModule time and first three results:"
	print "%fs, %s" % (secs, str(x_gpu.get()[:3]))
	print 'x:       ', self.x[self.Array_Size-1]
	print 'Func(x): ', x_gpu.get()[self.Array_Size-1],'Actual: ',round(abs(self.cx_p[self.Array_Size-1]-self.patch_x)+abs(self.cy_p[self.Array_Size-1]-self.patch_y)+abs(self.cz_p[self.Array_Size-1]-self.patch_z))
	self.Distance_Vector=np.column_stack((self.patchid,x_gpu.get()))
	
    def Magnify(self,patchidss):
	if patchidss == []:
		self.parent.Transp.setValue(5)
		return
	self.patch_x=self.patch_y=self.patch_z=0
	neighbours=[]
	patch_coord_size=0
	level_highlight=0
	for ids,value in zip(self.patchid, self.values):
		cx, cy, cz, sx, sy, sz, level, owner, avghops, maxhops, minhops= self.parent.agent.patch_coords_dict[ids]
		if ids in patchidss:
			level_highlight=level
			self.patch_x=cx
			self.patch_y=cy
			self.patch_z=cz
			break
	
	if level_highlight == 0:
			tolerance_rate=level_highlight+20
	elif level_highlight == 1: 	
			tolerance_rate=level_highlight+15
	elif level_highlight == 2: 
			tolerance_rate=level_highlight+10
#Parallel application of the finding the distance between patches and patch of interest
	start = cuda.Event()
	end = cuda.Event()
	start.record()
 	self.Cuda_Distance()
	end.record() 
	end.synchronize()
	secs = start.time_till(end)*1e-3
	print "parallel %fs" % (secs)
	start = cuda.Event()
	end = cuda.Event()
	start.record()
	for ids,value in zip(self.patchid, self.values):
		cx, cy, cz, sx, sy, sz, level, owner, avghops, maxhops, minhops= self.parent.agent.patch_coords_dict[ids]

     	        for new_ids in patchidss:
			if ids == new_ids:
				continue 
		Difference_x=abs(self.patch_x-cx)
		Difference_y=abs(self.patch_y-cy)
		Difference_z=abs(self.patch_z-cz)	
	
		if Difference_x < tolerance_rate and Difference_y < tolerance_rate and Difference_z < tolerance_rate:
			neighbours.append(ids)
	end.record() 
	end.synchronize()
	secs = start.time_till(end)*1e-3
	print "serial %fs" % (secs)

	return neighbours

    def crop(self,image, crop_criteria):
	    """ Crops the transparent background pixels out of a QImage and returns the
	    result.
	    """
	    def min_x(image):
		for x in range(image.width()):
		    for y in range(image.height()):
			if not crop_criteria(image.pixel(x,y)): return x

	    def max_x(image):
		for x in range(image.width()-1, -1, -1):
		    for y in range(image.height()):
			if not crop_criteria(image.pixel(x,y)): return x+1

	    def min_y(image):
		for y in range(image.height()):
		    for x in range(image.width()):
			if not crop_criteria(image.pixel(x,y)): return y

	    def max_y(image):
		for y in range(image.height()-1, -1, -1):
		    for x in range(image.width()-1, -1, -1):
			if not crop_criteria(image.pixel(x,y)): return y+1

	    mx, Mx = min_x(image), max_x(image)
	    my, My = min_y(image), max_y(image)
	    return image.copy(mx, my, Mx - mx, My - my)

    def saveImage(self, transparent):
	name, selectedFilter = QFileDialog.getSaveFileName(
	    self, "Save Image", "Patch-image.png", filter="*.png")
	if name:
	    image = self.grabFrameBuffer(withAlpha=transparent)
	    if transparent:
		image = self.crop(image, lambda p: not qAlpha(p))
	    else:
		image = self.crop(image, lambda p: qGray(p) == 255)
	    image.save(name)

    def variable_implode(self):
    	print "Idea MAannnn"
    	self.patch_size=0
    	for i in range(40):
			self.patch_size+=.01
			self.patch_size=self.patch_size % .7
			print self.patch_size
			self.PlaneUpdate()
			sleep(0.25)	

    def keyReleaseEvent(self, event):
	super(Patch3dGLWidget, self).keyReleaseEvent(event)
	# This adds the ability to save an image file if you hit 'p' while the
	# viewer is running.
	print "******************************METRICS*************************************\n"
	if event.key() == Qt.Key_Z:
	    self.saveImage(False)
	if event.key() == '\033':
		print 'Closing..'
		self.destroy_PBOs()
		self.destroy_texture()
		exit()
	if event.key() == Qt.Key_E:	
		print 'toggling cuda'
		self.enable_cuda = not self.enable_cuda
	elif event.key() == Qt.Key_L:
	    print "Idea of a lifetime"
	    self.variable_implode()
	elif event.key() == Qt.Key_X:
	    self.saveImage(True)
	elif event.key() == Qt.Key_P:
	    
	    print "Processing Highlights here:\n"
	    #profile.runctx('self.parent.agent.processHighlights()', globals(), locals())
	    print "PLane Update() here:\n"
	    #profile.runctx('self.PlaneUpdate()', globals(), locals())
	    print "PaintGL here:\n"
	    #profile.runctx('self.paintGL()', globals(), locals())
	    print "Patch Color Bar here:\n"
	    #profile.runctx('self.drawPatchColorBar()', globals(), locals())
	elif event.key() == Qt.Key_R:
	    print "Range:\n"
	    profile.runctx('self.Range()', globals(), locals())
	    print "\nRegisterRun:\n"
	    profile.runctx('self.parent.agent.registerRun(self.parent.agent.run_var)', globals(), locals())
	elif event.key() == Qt.Key_D:
	    print "\ndoLegend:\n"
	    #profile.runctx('self.doLegend()', globals(), locals())
	    print "drawCubes():"
	    #profile.runctx('self.drawCubes()', globals(), locals())
	    print "\nDrawAxis\n:"
	    #profile.runctx('self.drawAxis()', globals(), locals())
	elif event.key() == Qt.Key_K:
	    print "k"
	elif event.key() == Qt.Key_C:
	    print "Calculate FPS:\n"
	    #profile.runctx('self.calculateFPS()', globals(), locals())
	    print "\nChange PLane _ X:\n"
	    #profile.runctx('self.changePlane_x(self.PLane_state_x)', globals(), locals())
	    print "\nChange PLane _ Y:\n"
	    #profile.runctx('self.changePlane_y(self.PLane_state_y)', globals(), locals())
	    print "\nChange PLane _ Z:\n"
	    #profile.runctx('self.changePlane_z(self.PLane_state_z)', globals(), locals())
	    print "\nChange Title:\n"
	    #profile.runctx('self.changeTitle(self.trans)', globals(), locals())
	    print "\nChange Value for Slider:\n"
	    #profile.runctx('self.changeValue_for_Level_Slider(self.input_level)', globals(), locals())
	    print "\nChange Value for Transparency:\n"
	    #profile.runctx('self.changeValue_for_Level_Slider(self.transFact)', globals(), locals())		
	elif event.key() == Qt.Key_M:
	    print "Magnify:\n"
	    profile.runctx('self.Magnify(self.highlight_ids)', globals(), locals())
	    print "\nMagnifymode:\n"
	    profile.runctx('self.Magnifymode(self.magnify_mode)', globals(), locals())
	elif event.key() == Qt.Key_1:
	    self.patch_size+=.05
	    self.setPatchSize(self.patch_size)
	elif event.key() == Qt.Key_2:
	    self.patch_size-=.05
	    self.setPatchSize(self.patch_size)
	elif event.key() == Qt.Key_N:
	    print "\nneighbour_Change mode:\n"
	    #profile.runctx('self.neighbour_Change(self.highlight_ids)', globals(), locals())
	elif event.key() == Qt.Key_H:
	    print "highlighlightDrawing():"
	    #profile.runctx('self.highlightDrawing()', globals(), locals())
	elif event.key() == Qt.Key_Q:
	    print self.rotation
	elif event.key() == Qt.Key_R:
	    print self.rotation
	elif event.key() == Qt.Key_5:
	    self.input_level=0
	    self.changeValue_for_Level_Slider(self.input_level)
	elif event.key() == Qt.Key_0:
	    self.propagate=True
	    self.changeValue_for_propagate()
	elif event.key() == Qt.Key_6:
	    self.input_level=1
	    self.changeValue_for_Level_Slider(self.input_level)
	elif event.key() == Qt.Key_7:
	    self.input_level=2
            self.changeValue_for_Level_Slider(self.input_level)
	elif event.key() == Qt.Key_3:
	    self.transFact+=.1
	    self.parent.Transp.setValue(self.transFact)
	    self.repaint()
	    #self.changeValue_for_transparancy(self.transFact)
	elif event.key() == Qt.Key_4:
	    self.transFact-=.1
	    self.parent.Transp.setValue(self.transFact)
	    self.repaint()
	    #self.changeValue_for_transparancy(self.transFact)
	elif event.key() == Qt.Key_S:
	    print "Set PatchSize\n"
	    #profile.runctx('self.setPatchSize(self.patch_size)', globals(), locals())		
	    print "Selection changed:"
	    #profile.runctx('self.parent.agent.selectionChanged(self.highlight_ids)', globals(), locals())
	elif event.key() == Qt.Key_U:
	    print "UpdateHighlights\n"
	    #profile.runctx('self.updateHighlights(self.highlight_ids)', globals(), locals())	
	elif event.key() == Qt.Key_W:
	    self.Wire_Frame=not self.Wire_Frame
	    if not self.highlight_ids == []:
		self.highlightList.update()
		self.repaint()
	    else:
		self.cubeList.update()
		self.repaint()
	    sum1 = summary.summarize(self.neighbours)
	    print "Self.neighbours_dict(Neighbours/Magnify)",summary.print_(sum1) 
	    sum1 = summary.summarize(self.parent.agent.patch_coords_dict)
	    print "Patch_dict\n",summary.print_(sum1)
	    sum1 = summary.summarize(self.Distance_Vector)
	    print "self.PlaneValues\n",summary.print_(sum1)
	    sum1 = summary.summarize(self.x_colors)
	    print "self.highlight_ids\n",summary.print_(sum1)
	    sum1 = summary.summarize(self.cubeList)
	    #DddDprint "self.\n",summary.print_(sum1)
	    sum1 = summary.summarize(self.AxisList)
	elif event.key() == Qt.Key_A:
	    print "repainting"
	    if not self.highlight_ids == []:
		self.highlightList.update()
	    else:
		self.cubeList.update()
	#print "\n************************************************************"
    def create_PBOs(self,w,h):
	    num_texels = w*h
	    data = np.zeros((num_texels,4),np.uint8)
	    self.source_pbo = glGenBuffers(1)
	    glBindBuffer(GL_ARRAY_BUFFER, self.source_pbo)
	    glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
	    glBindBuffer(GL_ARRAY_BUFFER, 0)
	    self.pycuda_source_pbo = self.cuda_gl.BufferObject(long(self.source_pbo))
	    self.dest_pbo = glGenBuffers(1)
	    glBindBuffer(GL_ARRAY_BUFFER, self.dest_pbo)
	    glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
	    glBindBuffer(GL_ARRAY_BUFFER, 0)
	    self.pycuda_dest_pbo = self.cuda_gl.BufferObject(long(self.dest_pbo))
	    print self.pycuda_dest_pbo
    def destroy_PBOs(self):
	    for pbo in [self.source_pbo, self.dest_pbo]:
		glBindBuffer(GL_ARRAY_BUFFER, long(pbo))
		glDeleteBuffers(1, long(pbo));
		glBindBuffer(GL_ARRAY_BUFFER, 0)
	    self.source_pbo,self.dest_pbo,self.pycuda_source_pbo,self.pycuda_dest_pbo = [None]*4

    def create_texture(self,w,h):
	    self.output_texture = glGenTextures(1)
	    glBindTexture(GL_TEXTURE_2D, self.output_texture)
	    # set basic parameters
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	    # buffer data
	    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
			 w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

    def destroy_texture(self):
	    glDeleteTextures(self.output_texture);
	    output_texture = None

	    def Pixel_Buffer_Object(self):
		print "hello"
		#print "I am in pixel Buffer Object"
# Processes the planes in the planeValues list 
    def Process_planes(self):	
		self.PlaneValues.insert(1,False)
		self.PlaneValues.insert(2,False)
		self.PlaneValues.insert(3,False)
		self.PlaneValues.insert(4,False)
		self.PlaneValues.insert(5,False)
		self.PlaneValues.insert(6,False)
		self.PlaneValues.insert(7,False)
		self.PlaneValues.insert(8,False)		
		self.PlaneValues.insert(9,False)
		self.PlaneValues.insert(10,False)
		self.PlaneValues.insert(11,False)
		self.PlaneValues.insert(12,False)

#With a given Patch Id Finds all the neighbours in the vicinity and returns the patch_ids of the neighbours.It uses the help of distance from the patch of interest.

    def neighbour_Change(self,patchidss):
	if patchidss == []:
		return

	patch_x=patch_y=patch_z=0
	neighbours=[]
	patch_coord_size=0
	level_highlight=0
	#tolerance_rate=2
	#print patchidss

	for ids,value in zip(self.patchid, self.values):
		cx, cy, cz, sx, sy, sz, level, owner, avghops, maxhops, minhops= self.parent.agent.patch_coords_dict[ids]
		if ids in patchidss:
			level_highlight=level
			patch_x=cx
			patch_y=cy
			patch_z=cz
			break
		else: 
			continue


	if level_highlight == 0:
			tolerance_rate=9
	elif level_highlight == 1: 	
			tolerance_rate=7
	elif level_highlight == 2: 
			tolerance_rate=5

	for ids,value in zip(self.patchid, self.values):
		cx, cy, cz, sx, sy, sz, level, owner, avghops, maxhops, minhops= self.parent.agent.patch_coords_dict[ids]
		if self.PlaneValues[0] == True:		
			if cx < (self.X_interval):
					continue
		if self.PlaneValues[1] == True:
				if cx < (self.X_interval*2):
					continue
		if self.PlaneValues[2] == True:
				if cx > (self.X_interval*3):
					continue
		if self.PlaneValues[3] == True:
				if cx > (self.X_interval*4):
					continue
		if self.PlaneValues[4] == True:		
			if cy < (self.Y_interval):
					continue
		if self.PlaneValues[5] == True:
				if cy < (self.Y_interval*2):
					continue
		if self.PlaneValues[6] == True:
				if cy > (self.Y_interval*3):
					continue
		if self.PlaneValues[7] == True:
				if cy > (self.Y_interval*4):
					continue
		if self.PlaneValues[8] == True:		
			if cz < (self.Z_interval):
					continue
		if self.PlaneValues[9] == True:
				if cz < (self.Z_interval*2):
					continue
		if self.PlaneValues[10] == True:
				if cz > (self.Z_interval*3):
					continue
		if self.PlaneValues[11] == True:
				if cz > (self.Z_interval*4):
					continue

		if ids in patchidss:
			continue 
		Difference_x=abs(patch_x-cx)
		Difference_y=abs(patch_y-cy)
		Difference_z=abs(patch_z-cz)		
		Distance=int(round(Difference_x+Difference_y+Difference_z))
		#Distance=sqrt(pow(Difference_x,2)+pow(Difference_y,2)+pow(Difference_z,2))
	
		if Distance < tolerance_rate:
			neighbours.append(ids)

	return neighbours

    def drawPatchColorBar(self, x = 20, y = 90, w = 20, h = 120):
	"""Draw the color bar for nodes."""
	patch_bar = []
	for i in range(11):
	    patch_bar.append(self.patch_cmap.getColor(float(i)/10.0))

	drawGLColorBar(patch_bar, x, y, w, h, "P")

#Function gets called when the there are values in the highlightids
    @Slot(list)
    def updateHighlights(self, patchids):
	"""Given a list of the patch ids to be highlighted, changes
	   the alpha values accordingly and notifies listeners.
	"""
	c1=0 	
	if not self.Highlight_from_other_Modules:
		self.Highlight_from_other_Modules=self.parent.agent.patch_highlights
	if self.parent.agent.patch_highlights:
		self.Highlight_from_other_Modules=self.parent.agent.patch_highlights
		#print "in updateHiighlights",self.parent.agent.Ids_from_other_modules
		#print type(self.Highlight_from_other_Modules)
		self.new=[]
		#print self.parent.agent.Ids_from_other_modules[2]
		i=0
		for ids in self.Highlight_from_other_Modules:
			self.new.append(self.parent.agent.Ids_from_other_modules[i])
			i=i+1
			#print i

		#print self.new
		#print type(self.new)
				#print self.Highlight_from_other_Modules[3],"in updatehigh"

  	if patchids: # Alpha based on appearance in these lists
	    #self.set_all_alphas(0.2)
	    for patch in patchids:
		c1=c1+1
	if (c1<2):
		if patchids: # Alpha based on appearance in these lists
		    #self.set_all_alphas(0.2)
		    for patch in patchids:
			cx, cy, cz, sx, sy, sz, level, owner, avghops, maxhops, minhops= self.parent.agent.patch_coords_dict[patch]
	 		#print "cx : ",cx,"\ncy : ",cy,"\ncz : ",cz,"\nsx : ",sx,"\nsy : ",sy,"\nsz : ",sz,"\nlevel :",level,"\nPatch  id :",patch  
	      		self.initPatch(cx,cy,cz,sx,sy,sz,level, owner, avghops, maxhops, minhops) 

	self.highlight_ids.append(patchids)
	self.PlaneUpdate()
	if self.magnify_mode:
		self.parent.Transp.setValue(0)
	if self.magnify_mode:
		self.neighbours=self.Magnify(patchids)
		mode="Magnifying..."
	else:
		self.neighbours=self.neighbour_Change(patchids)
		mode="Neighboring"

	#print "Mode: ",mode,"\n ",self.neighbours 
	#self.updateDrawing(patchids)


#Function gets called when the Spread slider value is changed

    def changeValue_for_Spread(self, value):
	     
	self.BlowuP=value
	self.updateGL()

#Function gets called when the opacity slider vaalue is changed

    def changeValue_for_transparancy(self, value):
	     
	self.transFact=value
	print self.transFact
	if not self.highlight_ids == []:
		self.highlightList.update()
	else:
		self.cubeList.update()
	self.repaint()
	self.updateGL()  

#Function gets called when the Level slider is changed 

    def changeValue_for_Level_Slider(self, value):
	self.input_level=value
	self.Range()
	if not self.highlight_ids == []:
		self.highlightList.update()
	else:
		self.cubeList.update()
	self.updateGL()

    def paintGL(self):
       
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	self.orient_scene()
	if self.values is None:
	    return
	super(Patch3dGLWidget, self).paintGL()
	self.doAxis()
	if not self.highlight_ids == []: 
		self.highlightList()	
	else:
		self.cubeList()

	self.calculateFPS()
	#print self.fps
	with overlays2D(self.width(), self.height(), self.bg_color):
		self.drawPatchColorBar()     	

#responsible for the implsion operation 
    def setPatchSize(self, patch_size):
  	self.patch_size = patch_size   
	if not self.highlight_ids == []:
		self.highlightList.update()
	else:
		self.cubeList.update()
	self.doAxis()	 
	self.calculateFPS() 
	self.updateGL()

#Code from Torus module of the patch 

#Inclusion of the axis in the rendering scene
    def drawAxis(self):
	"""This function does the actual drawing of the lines in the axis."""
	glLineWidth(2.0)
	with glSection(GL_LINES):
	    glColor4f(1.0, 0.0, 0.0, 1.0)
	    glVertex3f(0, 0, 0)
	    glVertex3f(self.axisLength, 0, 0)
	    glColor4f(0.0, 1.0, 0.0, 1.0)
	    glVertex3f(0, 0, 0)
	    glVertex3f(0, -self.axisLength, 0)
	    glColor4f(0.0, 0.0, 1.0, 1.0)
	    glVertex3f(0, 0, 0)
	    glVertex3f(0, 0, -self.axisLength)
	    


#Code from Torus module of the patch 
#Inclusion of the axis in the rendering scene
    def doAxis(self):
	"""This function does the actual drawing of the lines in the axis."""
	glViewport(0,0,80,80)

	glPushMatrix()
	with attributes(GL_CURRENT_BIT, GL_LINE_BIT):
	    glLoadIdentity()
	    glTranslatef(0,0, -self.axisLength)
	    glMultMatrixd(self.rotation)
	    with disabled(GL_DEPTH_TEST):
		self.axisList()

	glPopMatrix()
	glViewport(0, 0, self.width(), self.height())

"""
Code from Torus module of the patch 
"""
class PatchColorTab(GLColorTab):
    """Color controls for Patch views."""

    def __init__(self, parent, mframe):
	"""Create the 3dColorTab."""
	super(PatchColorTab, self).__init__(parent, mframe)

    def createContent(self):
	"""Overriden createContent adds
	   to any existing ones in the superclass.
	"""
     # super(Torus3dColorTab, self).createContent()
	self.layout.addSpacerItem(QSpacerItem(5,5))
	self.layout.addWidget(self.buildColorMapWidget("Patch Colors",
	    self.colorMapChanged, "patches"))
	self.layout.addWidget(self.buildBoxSlider())


    @Slot(ColorMap, str)
    def colorMapChanged(self, color_map, tag):
	"""Handles change events from the Patch controls."""
	scene = self.mframe.agent.requestScene(tag)
	scene.color_map = color_map
	print color_map
	scene.processed = False
	scene.announceChange()

    def buildBoxSlider(self):
	widget = QWidget()

	layout2 = QVBoxLayout()

	layout = QHBoxLayout()
	label = QLabel("Patch size: ")
	label1 = QLabel("Implosion")

	layout.addWidget(label)
	layout.addItem(QSpacerItem(5,5))


	self.boxslider = QSlider(Qt.Horizontal, widget)

	self.boxslider.setRange(0,10)
	self.boxslider.setSliderPosition(
	    int(8 * self.mframe.glview.patch_size))
	self.boxslider.setTickInterval(1)

	self.boxslider.valueChanged.connect(self.boxSliderChanged)

	layout2.addWidget(label1)   
	layout.addWidget(self.boxslider)
	layout2.addItem(QSpacerItem(5,5))

	layout2.addLayout(layout)
	widget.setLayout(layout2)

	return widget

    @Slot(int)
    def boxSliderChanged(self, value):

	patch_size = value / float(10)
	self.mframe.glview.patch_size = patch_size*.1
	self.mframe.agent.requestScene("patches").announceChange()
	self.mframe.glview.setPatchSize(patch_size)

	QApplication.processEvents()

    def buildColorMapWidget(self, title, fxn, tag):
	"""Integrates ColorMapWidgets into this Tab."""
	color_map = self.mframe.agent.requestScene(tag).color_map

	groupBox = QGroupBox(title, self)
	layout = QVBoxLayout()
	color_widget = ColorMapWidget(self, color_map, tag)
	color_widget.changeSignal.connect(fxn)

	layout.addWidget(color_widget)
	groupBox.setLayout(layout)
	return groupBox

    # Copied mostly from GLModule
    # TOp: Factor out this color widget builder to something reusable 
    # like the ColorMapWidgetDO
    def buildNodeDefaultColorWidget(self):
	"""Creates the controls for altering the patch default color
	   (when there is no node data).
	"""
	widget = QWidget()
	layout = QHBoxLayout()
	label = QLabel("Default (no data) Patch Color")
	self.nodeDefaultBox = ClickFrame(self, QFrame.Panel | QFrame.Sunken)
	self.nodeDefaultBox.setLineWidth(0)
	self.nodeDefaultBox.setMinimumHeight(12)
	self.nodeDefaultBox.setMinimumWidth(36)
	self.nodeDefaultBox.clicked.connect(self.nodeDefaultColorChange)

	# self.mframe.agent.module_scene.node_default_color)
	self.default_color = ColorMaps.gl_to_rgb(self.mframe.glview.default_node_color)
	self.nodeDefaultBox.setStyleSheet("QFrame {\n background-color: "\
	    + ColorMaps.rgbStylesheetString(self.default_color) + ";\n"
	    + "border: 1px solid black;\n border-radius: 2px;\n }")
	layout.addWidget(label)
	layout.addItem(QSpacerItem(5,5))
	layout.addWidget(self.nodeDefaultBox)
	widget.setLayout(layout)
	return widget
