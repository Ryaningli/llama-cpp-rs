# Fork 本地改动说明

> 本文件记录 fork 仓库相对于上游 `utilityai/llama-cpp-rs` 的所有改动，
> 以便主线更新时评估合并冲突和回归风险。

## 改动概览

共修改 6 个文件，新增 1 个 feature。

### 1. 新增 feature: `dynamic-backends-no-variants`

基于 `dynamic-backends`，区别是不设置 CMake 的 `GGML_CPU_ALL_VARIANTS=ON`，
只构建一个通用的 `libggml-cpu.so`，而不是为每种 CPU 微架构（haswell/zen4/...）各编译一份。

**修改文件链：**

| 文件 | 改动 |
|------|------|
| `llama-cpp-sys-2/Cargo.toml` | +1 行 feature 定义 |
| `llama-cpp-2/Cargo.toml` | +1 行 feature 透传 |
| `examples/simple/Cargo.toml` | +1 行 feature 透传 |

**逻辑修改：**

| 文件 | 改动 |
|------|------|
| `llama-cpp-sys-2/build.rs:830` | `GGML_CPU_ALL_VARIANTS` 条件化，`no-variants` 时不设置 |
| `llama-cpp-2/src/llama_backend.rs:186,193,204` | `#[cfg(feature = "dynamic-backends")]` → `#[cfg(any(feature = "dynamic-backends", feature = "dynamic-backends-no-variants"))]` |
| `examples/simple/src/main.rs:174` | 添加 `load_backends()` 调用（`dynamic-backends` 模式必须） |

### 2. Bug 修复: build.rs hard_link 竞态条件

`llama-cpp-sys-2/build.rs` 第 1108-1128 行：

原代码用 `if !dst.exists() { hard_link() }` 模式复制共享库到 `target/release/`，
在重复构建时 `exists()` 检查与 `hard_link()` 之间存在竞态，导致 `EEXIST` 错误。

修复方式：在 `hard_link` 前无条件 `remove_file`（忽略"文件不存在"错误）。

涉及 3 处相同模式（target 目录、examples、deps）。

## 合入注意事项

### 上游合并时可能冲突的位置

1. **`llama-cpp-sys-2/build.rs` 第 823-833 行**
   - `dynamic-backends` CMake 配置块
   - 如果上游修改了此区域的 CMake 参数，需要保留 `GGML_CPU_ALL_VARIANTS` 的条件判断

2. **`llama-cpp-sys-2/build.rs` 第 1108-1128 行**
   - hard_link 复制逻辑
   - 如果上游重构了这段代码（比如改用 `std::fs::copy`），合并时优先采用上游方式，
     但需确保也修复了 `EEXIST` 竞态问题

3. **`llama-cpp-2/src/llama_backend.rs` 第 186-209 行**
   - `#[cfg]` 条件
   - 如果上游增加了新的 `#[cfg(feature = "dynamic-backends")]` 项，也需要加上
     `any(feature = "dynamic-backends", feature = "dynamic-backends-no-variants")`

4. **`examples/simple/Cargo.toml` 和 `examples/simple/src/main.rs`**
   - 上游可能重构 example 结构或添加新 feature
   - 合并时确保 `dynamic-backends-no-variants` 透传和 `load_backends()` 调用不丢失

### 合入策略

```bash
# 1. 添加上游 remote（首次）
git remote add upstream https://github.com/utilityai/llama-cpp-rs.git

# 2. 拉取上游最新代码
git fetch upstream

# 3. 合并到当前分支
git merge upstream/main

# 4. 解决冲突时，参考上述"可能冲突的位置"
#    优先保留上游代码，重新应用我们的改动

# 5. 合并后重新构建验证
cargo clean -p llama-cpp-sys-2 --release
cargo clean -p llama-cpp-2 --release
cargo build --release -p simple --features dynamic-backends-no-variants
```

### 如果上游也修复了 hard_link 问题

删除我们对 `build.rs` 第 1108-1128 行的改动，使用上游的修复即可。

### 如果上游也添加了类似的 no-variants 功能

评估是否可以切换到上游的实现，删除我们的 `dynamic-backends-no-variants` feature。

## 运行命令

```bash
# 带 CPU 变体（原行为，构建多种微架构版本）
cargo build --release -p simple --features dynamic-backends

# 不带 CPU 变体（新增，只构建通用版本）
cargo build --release -p simple --features dynamic-backends-no-variants

# 运行（需设置 LD_LIBRARY_PATH）
LIB_DIR=$(ls -d target/release/build/llama-cpp-sys-2-*/out/lib | tail -1) \
BACKENDS_DIR=$(ls -d target/release/build/llama-cpp-sys-2-*/out/backends | tail -1) \
LD_LIBRARY_PATH="$LIB_DIR:$BACKENDS_DIR" \
./target/release/simple -p "你好" --n-len 64 local /path/to/model.gguf
```
