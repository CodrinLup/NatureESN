function err = errors(state, output)

up = (state*output')^2;
down = (norm(state)^2)*(norm(output)^2);
err = up/down;
end