import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeCardsComponent } from './home-cards/home-cards.component';
import { Ur5eVisualComponent } from './ur5e-visual/ur5e-visual.component';
import { Ur5eCommandComponent } from './ur5e-command/ur5e-command.component';

const routes: Routes = [
  { path: '', redirectTo: '/app-home-cards', pathMatch: 'full' },
  { path: 'app-home-cards', component: HomeCardsComponent},
  { path: 'ur5e-visual', component: Ur5eVisualComponent },
  { path: 'ur5e-command', component: Ur5eCommandComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
