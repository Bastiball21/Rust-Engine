# Makefile for Aether Chess Engine
# Automates building for Linux MUSL (static) and Windows MSVC.

# --- CONFIGURATION ---
# Base name from Cargo.toml (Aether_v0_0_1)
BIN_NAME := Aether_v0_0_1
VERSION := v0.0.1
RELEASE_DIR := C:\Users\Basti\Desktop\Rust-Chess-Engine\rust_engine\target\release
LINUX_MUSL_TARGET := x86_64-unknown-linux-musl
WINDOWS_MSVC_TARGET := x86_64-pc-windows-msvc
# ---------------------


# --- TARGETS ---

.PHONY: all clean build windows linux

# Default build: Builds for the native host (Windows MSVC in your case)
build:
	@echo "Building Aether for Windows Host (Native Release)..."
	cargo build --release

# Build for Windows MSVC (explicitly)
windows:
	@echo "Building Aether v$(VERSION) for Windows MSVC..."
	cargo build --target $(WINDOWS_MSVC_TARGET) --release
	@echo "Windows executable built successfully:" $(RELEASE_DIR)/$(WINDOWS_MSVC_TARGET)/$(BIN_NAME).exe

# Build for Linux MUSL (Static Linking)
# REQUIRES: The 'x86_64-unknown-linux-musl' Rust target and a cross-linker (like x86_64-w64-mingw32-gcc)
linux:
	@echo "Building Aether v$(VERSION) for Linux MUSL (Static Linking)..."
	cargo build --target $(LINUX_MUSL_TARGET) --release
	@echo "Linux executable built successfully:" target/$(LINUX_MUSL_TARGET)/release/$(BIN_NAME)

# Builds all configured targets
all: build linux windows

# Removes all compiled artifacts
clean:
	@echo "Cleaning compiled artifacts..."
	cargo clean