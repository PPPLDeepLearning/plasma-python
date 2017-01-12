from plasma.conf_parser import parameters
import os

conf = parameters(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'conf.yaml'))
