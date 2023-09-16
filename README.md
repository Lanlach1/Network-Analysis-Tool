# Network-Analysis-Tool

This readme will provide a descirption of how to properly use the node analysis and edge analysis tools provided in this repository.

This code relies on excel files and images to define the networks upon which analysis can be performed. In the "Example Networks" folder we have provided 5 examples of this data so that you may use it and understand how it works.

For both the edge and node anaylsis scripts, you only need to edit a handful of variables. 

USING THE EDGE ANALYSIS TOOL:
The only code you need to edit when using this tool is from line 16 to line 28. Please refer to the descriptions of each variable below for use!

node_path:
This variable conatins the path to the xls (or xlsx) file which provides the x-y coordinates of all nodes within your network. Please refer to the "nodes" xls (or xlsx) files in the "Example Networks" folder.

edge_path:
This variable is the path to the xls (or xlsx) file which indicates which nodes of your network are connected. Please refer to the "edges" xls (or xlsx) files in the "Example Networks" folder.

save_data: 
If set to True, the code will perform an edge analysis of the given network and return the MFPT data as an xlsx file to wherever you set the "save_data_path" variable.
If set to False, the code will NOT perform an analysis of the network but rather attempt to pull from an existing MFPT xlsx file. 
You may want to set the variable to False only after you created the MFPT xls file from running the code once with save_data = true. This can help avoid excess computation.

image_path:
If you extracted a network from an image and want to use that image as a backround, you can put the path of that image as this variable. 
In the "Example Networks" folder, both the Neuromuscular Junction and S2 Cell data have an image. You can use these example networks for reference.
If you do not wish to use a backround image for your network, you can leave the image_path variable as "None".

image_bounds:
This variable sets the x-y bounds of the image backround (should you choose to use one) relative to the x-y positions of the network nodes.
For the bounds of your image, this variabnle is of the form [Min x, Max x, Min y, Max y].
If using the S2 Cell or Neuromuscular junction data sets in the "Example Networks" folder. Please use the bounds defined in the README for each network.

name:
just the name of the network you are using, when you save the analysis figure that is generated, it will use that as a name for the file.

line_width:
Defines the width of the lines in the analysis figure generated. You may have to play around with this a little until the figure looks good. You generally want thicker lines for less dense networks and thinner for more dense networks.

save_graph:
If true, the figure generated will save to save_graph_path in the form of both a PDF and PNG.
If false, the figure will not save anywhere, you can still do it manually though. 

save_graph_path:
The path in which the figures generated will save.
WARNING!!! Dont specify a file as the path but only a folder to which you want to save the files.

colorbar_limits:
If you want to adjust where the colorbar reaches its max and min, use this variable! This variable is of the form [min, max]
You may want to play around with this a little until you get a good range for your data.

figure_dpi:
This variable will set the dpi of your figure. Totally unintuituve...


USING THE NODE ANALYSIS TOOL:
The only code you need to edit when using this tool is from line 16 to line 26. There variables are almost identical to the variables used for the edge analysis tool, with some SLIGHT changes. If the variable in question is not defined below, then you can use the definition from the Edge Analysis tool section of this README :D

line_color:
Sets the color of the lines used to connect the network in the figure. This ties to the "plot" function of matplotlib so only use key works that would work for that.
For example: "white", "red"....


