pub fn uci_println(line: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stdout(), "{}", line);
    let _ = std::io::stdout().flush();
}

pub fn uci_print(line: &str) {
    use std::io::Write;
    let _ = write!(std::io::stdout(), "{}", line);
    let _ = std::io::stdout().flush();
}
