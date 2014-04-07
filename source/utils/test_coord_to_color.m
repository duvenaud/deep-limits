% test coord to color

n_1d = 200;

xrange = linspace( -1, 1, n_1d );
[xa, xb] = ndgrid( xrange, xrange );
x = [ xa(:), xb(:) ];

%x = mod(x + 1 ,2) - 1;
x = sin(2.*pi.*x);

immat = NaN(size(x));
colors = coord_to_color2(x);
im = reshape(colors, n_1d, n_1d, 3);
imshow(im)