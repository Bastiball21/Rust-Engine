use log::LevelFilter;
use simplelog::{Config, WriteLogger};
use std::fs::File;

pub fn init_logging() {
    let _ = WriteLogger::init(
        LevelFilter::Info,
        Config::default(),
        File::create("aether.log").unwrap(),
    );
    log::info!("Logger initialized.");
}
