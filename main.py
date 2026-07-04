import meshreading as mr
import geometry as geo
import initialize as ini
import classconfig as cc
import solvesupple as ss

mr.read_mesh("fangdata.txt")

geo.geometry_main("output.txt")

ini.initialization_main()

ss.formvars_main()

ss.min_timestep()

ss.formIM()