/**************************************************************************/
/*  This file is part of POPCON.                                          */
/*                                                                        */
/*  Copyright (C) 2025                                                    */
/*    CEA (Commissariat à l'énergie atomique et aux énergies              */
/*         alternatives)                                                  */
/*                                                                        */
/*  you can redistribute it and/or modify it under the terms of the GNU   */
/*  Lesser General Public License as published by the Free Software       */
/*  Foundation, version 2.1.                                              */
/*                                                                        */
/*  It is distributed in the hope that it will be useful,                 */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of        */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         */
/*  GNU Lesser General Public License for more details.                   */
/*                                                                        */
/*  See the GNU Lesser General Public License version 2.1                 */
/*  for more details (enclosed in the file licenses/LGPLv2.1).            */
/*                                                                        */
/**************************************************************************/

//! Stat collections
use std::{
    collections::BTreeMap,
    fs::File,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
    time::Instant,
};
use tracing::field::Visit;
use tracing::Subscriber;
use tracing_subscriber::Layer;
use tracing_subscriber::{layer::Context, registry::LookupSpan};

struct StatsLayerInner {
    /// actual stats
    data: BTreeMap<String, serde_json::Value>,
    /// file to which it must be written
    file: File,
}

/// A subscriber that records span and events attributes when there is a stats attribute
/// Cloning returns a new refcounted reference to the same StatsLayer
#[derive(Clone)]
pub struct StatsLayer {
    /// None when already dumped once
    inner: Arc<Mutex<Option<StatsLayerInner>>>,
}

struct StatsVisitor<T: DerefMut + Deref<Target = StatsLayerInner>>(T);

fn i128_to_json(x: i128) -> serde_json::Number {
    if x < 0 {
        (x as i64).into()
    } else {
        (x as u64).into()
    }
}
/// lift x = f(x) from i128 to json values. Panics when something wrong happens.
fn do_integer(x: &mut serde_json::Value, mut f: impl FnMut(i128) -> i128) {
    match x {
        serde_json::Value::Number(n) => {
            let arg = if n.is_i64() {
                n.as_i64().unwrap() as i128
            } else if n.is_u64() {
                n.as_u64().unwrap() as i128
            } else {
                unreachable!("stats json should only contain integers")
            };
            *x = serde_json::Value::Number(i128_to_json(f(arg)));
        }
        _ => panic!("stats: tried to accumulate into {:?}", x),
    }
}

impl StatsLayerInner {
    fn record_int(&mut self, field_name: &str, value: i128) {
        let name = |suffix| format!("{}_{}", field_name, suffix);
        let jvalue = serde_json::Value::Number(i128_to_json(value));
        self.data.insert(name("last"), jvalue.clone());
        self.data.entry(name("first")).or_insert(jvalue.clone());
        self.data
            .entry(name("max"))
            .and_modify(|before| do_integer(before, |before| std::cmp::max(value, before)))
            .or_insert(jvalue.clone());
        self.data
            .entry(name("min"))
            .and_modify(|before| do_integer(before, |before| std::cmp::min(value, before)))
            .or_insert(jvalue.clone());
        self.data
            .entry(name("sum"))
            .and_modify(|before| do_integer(before, |before| before + value))
            .or_insert(jvalue.clone());
        self.data
            .entry(name("count"))
            .and_modify(|before| do_integer(before, |before| before + 1))
            .or_insert(serde_json::Value::from(1u64));
    }
}

impl<T: DerefMut + Deref<Target = StatsLayerInner>> StatsVisitor<T> {
    fn record(&mut self, field: &tracing::field::Field, value: serde_json::Value) {
        match field.name() {
            "message" | "stats" => return,
            _ => (),
        }
        self.0.data.insert(field.name().to_owned(), value);
    }

    fn record_int(&mut self, field: &tracing::field::Field, value: i128) {
        match field.name() {
            "message" | "stats" => return,
            _ => (),
        }
        self.0.record_int(field.name(), value);
    }
}

impl<T: DerefMut + Deref<Target = StatsLayerInner>> Visit for StatsVisitor<T> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.record(field, serde_json::Value::String(format!("{:?}", value)))
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.record(field, value.into())
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.record_int(field, value.into())
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.record(field, serde_json::Value::String(value.to_owned()))
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.record(field, serde_json::Value::Bool(value))
    }
}

impl StatsLayer {
    /// Creates a StatsSubscriber, writing to file when dump() is called.
    pub fn new(file: File) -> Self {
        let inner = StatsLayerInner {
            file,
            data: BTreeMap::new(),
        };
        Self {
            inner: Arc::new(Mutex::new(Some(inner))),
        }
    }

    /// Writes stats to a file. Noop after first invocation.
    pub fn dump(&self) {
        let mut guard = self.inner.lock().expect("poisoned lock");
        let inner = guard.take();
        drop(guard);
        if let Some(inner) = inner {
            match serde_json::to_writer(inner.file, &inner.data) {
                Ok(()) => {}
                Err(e) => tracing::warn!("failed to write stats: {}", e),
            };
        }
    }
}

fn has_field(m: &tracing::Metadata, name: &'static str) -> bool {
    m.fields().field(name).is_some()
}

struct SpanStartTime(Instant);

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for StatsLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        if !has_field(event.metadata(), "stats") {
            return;
        }
        let mut guard = self.inner.lock().expect("poisoned lock");
        if let Some(inner) = guard.as_mut() {
            let mut v = StatsVisitor(inner);
            event.record(&mut v);
        }
    }

    fn new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: Context<'_, S>,
    ) {
        if !has_field(attrs.metadata(), "timing") {
            return;
        }
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(SpanStartTime(Instant::now()))
        }
    }

    fn on_exit(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if let Some(SpanStartTime(start)) = span.extensions().get() {
                let time = start.elapsed().as_millis(); // max a few million years
                let key = format!("time_{}_ms", span.name());
                let mut guard = self.inner.lock().expect("poisoned lock");
                if let Some(inner) = guard.as_mut() {
                    inner.record_int(&key, time as i128);
                }
            }
        }
    }
}
