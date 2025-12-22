use log::LevelFilter;
use simplelog::{Config, WriteLogger};
use std::fs::File;

pub fn init_logging() {
    // If we fail to create the log file, we shouldn't panic, just fallback to no logging
    if let Ok(file) = File::create("aether.log") {
        let _ = WriteLogger::init(
            LevelFilter::Info,
            Config::default(),
            file,
        );
        log::info!("Logger initialized.");
    }
}
