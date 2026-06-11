CC := gcc
TARGET := main

SRCS := game.c main.c memory.c network.c tensor.c
OBJS := $(SRCS:.c=.o)
DEPS := $(SRCS:.c=.d)

CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -O3
DFLAGS := -std=c11 -Wall -Wextra -Wpedantic -O0 -DDEBUG -g

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $@..."
	@$(CC) $(CFLAGS) -o $@ $^
	@echo "Build complete: $@"

debug: $(OBJS)
	@echo "Linking $@..."
	@$(CC) $(DFLAGS) -o $@ $^
	@echo "Build complete: $@"

%.o: %.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(OBJS) $(DEPS) $(TARGET)
	@rm -rf *.dSYM
	@echo "Clean complete"

.PHONY: all clean debug run
