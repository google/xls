# DSLX Language Server

Many popular editors in the modern era are speaking a common protocol in order
to understand how to display, traverse, and maniulate languages: the
["Language Server Protocol"](https://en.wikipedia.org/wiki/Language_Server_Protocol).
This allows novel languages and DSLs, like DSLX, to expose a developer
experience integrated in their preferred editors and IDEs.

**To use the language server protocol in your editor you do not need to know any
details about the language server protocol.**

Language server feedback in the editor is useful to folks who are learning DSLX
as well as those developing in it on a regular basis! The language server
currently offers functionality such as:

*   Go-to-definition
*   Errors/warnings as you type
*   An "overview" of the symbols defined in a module

For more background on what the language server protocol can do, see
[the Language Server Protocol documentation and specification](https://microsoft.github.io/language-server-protocol/).

## Building the DSLX Language Server binary

The following are instructions for building the DSLX language server binary. By
a) placing the language server binary into your `PATH` and b) configuring your
editor to handle `.x` files by using it.

Follow the
[XLS build setup instructions](https://google.github.io/xls/#building-from-source)
so that the prerequisites are available for building binaries via Bazel. Then,
build the following `dslx_ls` binary and place it in your `PATH`:

```sh
$ bazel build -c opt //xls/dslx/lsp:dslx_ls
$ mkdir ~/bin/
$ cp -iv bazel-bin/xls/dslx/lsp/dslx_ls ~/bin/
$ export PATH=$HOME/bin:$PATH
```

Now that the language server binary is available in your `PATH`, you must
configure your editor to find/use it for `.x` files.

## Vim

First we must install `vim-plug` -- follow the latest instructions at
`https://github.com/junegunn/vim-plug`; e.g.:

```sh
$ curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
```

Add the following configuration to your `$HOME/.vimrc`:

```vim
call plug#begin()
Plug 'prabirshrestha/vim-lsp'
Plug 'prabirshrestha/asyncomplete-lsp.vim'
Plug 'mattn/vim-lsp-settings'

call plug#end()

let g:lsp_log_verbose = 1
let g:lsp_log_file = expand('~/vim-lsp.log')

if executable('dslx_ls')
    au User lsp_setup call lsp#register_server({
        \ 'name': 'dslx_ls',
        \ 'cmd': {server_info->['dslx_ls']},
        \ 'allowlist': ['dslx', '.x'],
        \ })
endif

let g:lsp_diagnostics_echo_cursor = 1
let g:lsp_diagnostics_highlights_enabled = 1
let g:lsp_diagnostics_signs_enabled = 1

au BufRead,BufNewFile *.x set filetype=dslx
```

Start Vim and execute `:PlugInstall` to get the new LSP plugins. After that
completes successfully, quit Vim (via `:q`).

Open `foo.x` in Vim and then paste the following contents:

```
fn main() -> u8 { u8:256 }
```

The following should show in the display line:

```
LSP: uN[8] Value '256' does not fit in the bitwidth of a uN[8] (8). Valid values are [0, 255].
```

If not, try using `:LspStatus` To see if any diagnostics are available.

If you correct the `u8` value to be 255 (and thus in range) you can run:

`:LspDocumentSymbol`

To see the defined symbol listing -- this shows all defined symbols in the file:

```
foo.x|1 col 1| method : main
```

### Troubleshooting

With the above `.vimrc` contents, logs should show up in `$HOME/vim-lsp.log`.

Issues can be filed against
[https://github.com/google/xls/issues](https://github.com/google/xls/issues)
with associated contents/logs.

## Emacs

The following `.emacsrc` snippet wires up the language server in emacs,
piggy-backing on the Rust major mode.

```elisp
(which-key-mode)
(require 'lsp-mode)
(add-to-list 'auto-mode-alist '("\\.x\\'" . rust-mode))
;; DSLX can make rust-mode very slow for large files due to angle bracket
;; matching which is inefficiently implemented. Disable the feature.
(setq rust-match-angle-brackets nil)
(add-to-list 'lsp-language-id-configuration '(rust-mode . "dslx"))
(lsp-register-client
 (make-lsp-client :new-connection (lsp-stdio-connection "~/bin/dslx_ls")
                  :major-modes '(rust-mode)
                  :server-id 'dslx-ls))
(add-hook 'rust-mode-hook 'lsp)
```

Additional details on Emacs enablement may be available in the
[Verible editor hook-up documentation](https://github.com/chipsalliance/verible/blob/master/verilog/tools/ls/README.md#hooking-up-to-editor).

## Sublime Text

### Create a DSLX syntax

Go to `Tools > Developer > New Syntax`, then replace values in the template:

*   Change the `file_extensions` value to be `x`
*   Change scope to be `source.dslx`
*   Replace all `example-c` in the file with `dslx`

Now when you open a `.x` file it should show the syntax in the lower right-hand
corner as `dslx`.

### Install LSP package

Instructions for installing the LSP package are given in the
[Verible documentation](https://github.com/chipsalliance/verible/blob/master/verilog/tools/ls/README.md#sublime).

### Add DSLX to LSP settings

Open the configuration via `Preferences > Package Settings > LSP > Settings` and
add the following client:

```
{
    "clients": {
        "dslx_ls": {
          "command": ["dslx_ls"],
          "enabled": true,
          "selector": "source.dslx"
        }
    }
}
```

Now open `foo.x` and paste in:

`fn main() -> u8 { u8:256 }`

There should be a red squiggle under the number `256` indicating that the value
is out of range for a `u8`.

## Other Editors

For non-Vim editors, see the instructions provided by
[our sister project, Verible](https://github.com/chipsalliance/verible/blob/master/verilog/tools/ls/README.md#hooking-up-to-editor).
