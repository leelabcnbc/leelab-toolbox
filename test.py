from leelabtoolbox import matlab


a = matlab.get_matlab_handle()

print(a.vl_version('verbose'))