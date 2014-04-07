string = '<page class="aqua"><h2>Avoiding Pathologies in Very Deep Networks<br> <small><span class="blue"> (Duvenaud, Rippel, Adams, Ghahramani 2014)</span></small></h2><center>  <img src="seed-0/latent_coord_map_layer_%d.png"></center></page>';
clc
for i = 1:100
    fprintf(string, i);
    fprintf('\n');
end