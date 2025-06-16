function weight = wei(state,output)
up = (state*output');
down = norm(output)^2;
weight = up/down;
end