# commit

格式：`<type>(<scope>): <subject>`

## type

说明commit的类别，使用以下标识（参考[commitlint](https://www.conventionalcommits.org/en/v1.0.0/)）

feat：新功能（feature）

fix：说明bug并修复bug

stage：说明bug但不（完全）修复bug，适用于多次提交。最终修复时使用fix

perf：优化（performance）

refactor：重构

docs：文档（documentation）

revert：版本回滚

style：风格

test：测试

chore：构建过程变动或辅助工具变动等

## scope(可选)

说明commit影响的范围，比如 level0, level1, 全局 等等，视项目不同而不同

## subject

简短描述，不超过50个字符。使用中文，结尾不加句号或其他标点符号
