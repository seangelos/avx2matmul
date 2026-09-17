#!/bin/bash

g++ -std=c++20 -g -O3 -Wall -Wextra -Wpedantic -ffast-math -march=native -lblis -o ./bin/Core ./main.cpp ./kernel.cpp
g++ -std=c++20 -g -O0 -Wall -Wextra -Wpedantic -ffast-math -march=native -lblis -o ./bin/CoreD ./main.cpp ./kernel.cpp

clang++ -stdlib=libc++ -std=c++20 -g -O3 -Wall -Wextra -Wpedantic -ffast-math -march=native -lblis -o ./bin/CoreC ./main.cpp ./kernel.cpp

objdump -Sls -M intel ./bin/Core > ./bin/Core.dis
objdump -Sls -M intel ./bin/CoreC > ./bin/CoreC.dis
