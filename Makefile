all: rand.ll #rand

rand.ll: rand.c
	clang -emit-llvm -S rand.c -o rand.ll

clean:
    rm -rf rand rand.ll

# Uncomment main function in rand.c to enable this
# rand: rand.c
# 	clang -O3 -Wall -o rand rand.c