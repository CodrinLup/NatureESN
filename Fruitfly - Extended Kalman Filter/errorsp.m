function err = errorsp(state, outputorig, outputmodif)

up = (state*outputmodif')^2;
down = (norm(state)^2)*(norm(outputorig)^2);
err = up/down;
end