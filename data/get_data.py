import MDSplus 
import numpy
import time
import sys

class gadata:
	"""GA Data Obj"""
	def __init__(self,signal,shot,tree=None,connection=None,nomds=False):

		# Save object values
		self.signal 		= signal
		self.shot   		= shot
		self.zdata              = -1
		self.xdata              = -1
		self.ydata 		= -1
		self.zunits             = ''
		self.xunits             = ''
		self.yunits		= ''
		self.rank 		= -1
		self.connection		= connection

		## Retrieve Data 
		t0 =  time.time()
		found = 0

		# Create the MDSplus connection (thin) if not passed in  
		if self.connection is None:
			self.connection = MDSplus.Connection('atlas.gat.com')

		# Retrieve data from MDSplus (thin)
		if nomds == False:
			try:     
				if tree != None:
					tag 	= self.signal
					fstree 	= tree
				else:					
					tag = self.connection.get('findsig("'+self.signal+'",_fstree)').value
  					fstree = self.connection.get('_fstree').value 
				self.connection.openTree(fstree,shot)
				self.zdata  	= self.connection.get('_s = '+tag).data()
				self.zunits 	= self.connection.get('units_of(_s)').data()  
				self.rank   	= numpy.ndim(self.zdata)	
				self.xdata     	= self.connection.get('dim_of(_s)').data()
				self.xunits 	= self.connection.get('units_of(dim_of(_s))').data()
				if self.xunits == '' or self.xunits == ' ': 
   					self.xunits     = self.connection.get('units(dim_of(_s))').data()
				if self.rank > 1:
					self.ydata 	= self.connection.get('dim_of(_s,1)').data()
                       			self.yunits 	= self.connection.get('units_of(dim_of(_s,1))').data()
					if self.yunits == '' or self.yunits == ' ':
 						self.yunits     = self.connection.get('units(dim_of(_s,1))').data()
				found = 1	

				# MDSplus seems to return 2-D arrays transposed.  Change them back.
				if numpy.ndim(self.zdata) == 2: self.zdata = numpy.transpose(self.zdata)
				if numpy.ndim(self.ydata) == 2: self.ydata = numpy.transpose(self.ydata)
				if numpy.ndim(self.xdata) == 2: self.xdata = numpy.transpose(self.xdata)

                	except Exception,e:
#				print '   Signal not in MDSplus: %s' % (signal,) 
				pass
		

		# Retrieve data from PTDATA
                if found == 0:
                        self.zdata = self.connection.get('_s = ptdata2("'+signal+'",'+str(shot)+')')
                        if len(self.zdata) != 1:
                                self.xdata = self.connection.get('dim_of(_s)')
                                self.rank = 1
                                found = 1

		# Retrieve data from Pseudo-pointname 
		if found == 0:
#			print '   Signal not in PTDATA: %s' % (signal,) 
			self.zdata = self.connection.get('_s = pseudo("'+signal+'",'+str(shot)+')')
			if len(self.zdata) != 1:
				self.xdata = self.connection.get('dim_of(_s)')
				self.rank = 1
			        found = 1

		if found == 0: 
#			print '   Signal not in pseudo-pointname: %s' % (signal,) 
			print "   No such signal: %s" % (signal,)
			return

                print '   GADATA Retrieval Time : ',time.time() - t0
