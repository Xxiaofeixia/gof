package Bio::Perl;

# LOFTEE 的 GRCh38 插件仍然包含 `use Bio::Perl;`，但当前代码路径并不调用
# Bio::Perl 中的函数。新版 bioperl 包不再提供这个兼容模块时，VEP 会在插件
# 编译阶段失败。这个最小 shim 只用于满足 LOFTEE 的加载依赖。

1;
