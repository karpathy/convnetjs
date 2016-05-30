var gulp = require('gulp');
var jshint = require('gulp-jshint');
var browserSync = require('browser-sync');
const reload = browserSync.reload;
gulp.task('copy', () =>
        gulp.src([
            'build/**/*'
        ], {
            base: './'
        }).pipe(gulp.dest('demo'))
);
// jshint files
gulp.task('jshint', function () {
    //gulp.src(['test/**/*.js'])
    //    .pipe(jshint())
    //    .pipe(jshint.reporter());
});
// start local http server for development
gulp.task('http-server', function () {
    browserSync({
        notify: false,
        // Customize the Browsersync console logging prefix
        logPrefix: 'WSK',
        // Allow scroll syncing across breakpoints
        scrollElementMapping: ['main', '.mdl-layout'],
        // Run as an https by uncommenting 'https: true'
        // Note: this uses an unsigned certificate which on first access
        //       will present a certificate warning in the browser.
        // https: true,
        server: ['.tmp', 'demo'],
        port: 3000
    });
    gulp.watch(['demo/**/*'], reload);
});
// start local http server with watch and livereload set up
gulp.task('server', function () {
    gulp.run('http-server');
});
gulp.task('default', function () {
    gulp.run('jshint', 'copy', 'server');
});

