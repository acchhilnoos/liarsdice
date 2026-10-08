CC := gcc
TARGET := main

SRCS := game.c main.c network.c play.c tensor.c train.c
OBJS := $(SRCS:.c=.o)
DEPS := $(SRCS:.c=.d)

CFLAGS := -std=c11 -Wall -Wextra -Wpedantic

all: CFLAGS += -O3 -march=native -ffast-math
all: $(TARGET)

debug: CFLAGS += -O0 -g -fsanitize=address
debug: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $@..."
	@$(CC) $(CFLAGS) -o $@ $^
	@echo "Build complete: $@"

%.o: %.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

test: CFLAGS += -O3
test: game.o network.o tensor.o test.o
	@echo "Linking $@..."
	@$(CC) $(CFLAGS) -o $@ $^ 2>> /dev/null
	@rm -f test.o

clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(OBJS) $(DEPS) $(TARGET) test
	@rm -rf *.dSYM
	@echo "Clean complete"

.PHONY: all clean debug run test
