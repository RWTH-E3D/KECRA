import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { Ur5eVisualComponent } from './ur5e-visual/ur5e-visual.component';
import { Ur5eCommandComponent } from './ur5e-command/ur5e-command.component';
import { AppHeaderComponent } from './app-header/app-header.component';
import { HomeCardsComponent } from './home-cards/home-cards.component';

import { FormsModule } from '@angular/forms';

@NgModule({
  declarations: [
    AppComponent,
    Ur5eVisualComponent,
    Ur5eCommandComponent,
    AppHeaderComponent,
    HomeCardsComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
